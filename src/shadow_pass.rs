use std::num::NonZeroU64;

use anyhow::Result;
use encase::{ShaderSize, ShaderType, UniformBuffer};
use nalgebra as na;

use crate::{
    camera::GpuCamera,
    gpu::Gpu,
    mesh::{Mesh, MeshVertexArrayType},
    phong_light::PhongLight,
    projection::wgpu_projection,
    scene::{GpuScene, Instance},
};

pub struct DirectionalShadowPass {
    splits: [f32; SPLIT_COUNT],
    pipeline: wgpu::RenderPipeline,
    pnuv_pipeline: wgpu::RenderPipeline,
    bg: wgpu::BindGroup,
    depth_tex: wgpu::Texture,
    proj_mat_buf: wgpu::Buffer,
    view_mat_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    out_bg: wgpu::BindGroup,
    out_bgl: wgpu::BindGroupLayout,
}

const MIN_UNIFORM_BUFFER_OFFSET_ALIGNMENT: u64 = 256;
const SPLIT_COUNT: usize = 3;
const SHADOW_MAP_SIZE: u32 = 2048;

#[derive(ShaderType)]
struct ShadowMapResult {
    num_splits: u32,
    #[align(16)]
    split_distances: [na::Vector4<f32>; 16],
}

fn calculate_frustum(
    view_mat: &na::Matrix4<f32>,
    proj_mat: &na::Matrix4<f32>,
) -> Result<[na::Point3<f32>; 8]> {
    let inv_projection_mat = proj_mat
        .try_inverse()
        .ok_or_else(|| anyhow::anyhow!("failed to invert projection mat"))?;
    let inv_view_mat = view_mat
        .try_inverse()
        .ok_or_else(|| anyhow::anyhow!("failed to invert camera mat"))?;

    let points = &[
        na::Point3::new(-1.0, -1.0, 0.0), // bottom-left near plane
        na::Point3::new(1.0, -1.0, 0.0),  // bottom-right near plane
        na::Point3::new(-1.0, 1.0, 0.0),  // top-left near plane
        na::Point3::new(1.0, 1.0, 0.0),   // top-right near plane
        na::Point3::new(-1.0, -1.0, 1.0), // bottom-left far plane
        na::Point3::new(1.0, -1.0, 1.0),  // bottom-right far plane
        na::Point3::new(-1.0, 1.0, 1.0),  // top-left far plane
        na::Point3::new(1.0, 1.0, 1.0),   // top-right far plane
    ];

    Ok(points.map(|p| {
        let p = inv_projection_mat * na::Vector4::new(p.x, p.y, p.z, 1.0);
        let p = p / p.w;
        let p = inv_view_mat * na::Vector4::new(p.x, p.y, p.z, 1.0);
        na::Point3::new(p.x, p.y, p.z)
    }))
}

fn split_frustum(
    frustum_points: &[na::Point3<f32>; 8],
    splits: &[f32],
) -> Vec<[na::Point3<f32>; 8]> {
    let [bln, brn, tln, trn, blf, brf, tlf, trf] = frustum_points;

    let bl = blf - bln;
    let br = brf - brn;
    let tl = tlf - tln;
    let tr = trf - trn;

    let mut result = Vec::with_capacity(splits.len());

    let mut frustum_split = *frustum_points;
    for split in splits.iter().copied() {
        frustum_split[4] = bln + bl * split;
        frustum_split[5] = brn + br * split;
        frustum_split[6] = tln + tl * split;
        frustum_split[7] = trn + tr * split;

        result.push(frustum_split);

        frustum_split[0] = frustum_split[4];
        frustum_split[1] = frustum_split[5];
        frustum_split[2] = frustum_split[6];
        frustum_split[3] = frustum_split[7];
    }

    result
}

impl DirectionalShadowPass {
    pub fn new(
        gpu: &Gpu,
        splits: [f32; SPLIT_COUNT],
        projection_mat: &na::Matrix4<f32>,
    ) -> Result<Self> {
        let depth_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: SPLIT_COUNT as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let shader = gpu.shader_from_file("./shaders/shadowMap.wgsl")?;
        let pnuv_shader = gpu.shader_from_file("./shaders/shadowMapPNUV.wgsl")?;

        let mat4_size: u64 = na::Matrix4::<f32>::SHADER_SIZE.into();
        let offset = mat4_size.max(MIN_UNIFORM_BUFFER_OFFSET_ALIGNMENT);

        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: NonZeroU64::new(offset),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: NonZeroU64::new(offset),
                        },
                        count: None,
                    },
                ],
            });

        let pipelinel = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

        let pnuv_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipelinel),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[
                        Mesh::pnuv_vertex_layout(),
                        Instance::pnuv_model_instance_layout(),
                    ],
                },
                fragment: None,
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipelinel),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[
                        Mesh::pn_vertex_layout(),
                        Instance::pn_model_instance_layout(),
                    ],
                },
                fragment: None,
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

        let view_mat_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: offset * splits.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let proj_mat_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: offset * splits.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &view_mat_buf,
                        offset: 0,
                        size: NonZeroU64::new(offset),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &proj_mat_buf,
                        offset: 0,
                        size: NonZeroU64::new(offset),
                    }),
                },
            ],
        });

        let out_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let near_far_ratio = (projection_mat[(2, 2)] + 1.0) / (projection_mat[(2, 2)] - 1.0);
        let z_near =
            (projection_mat[(2, 3)] * (near_far_ratio / 2.0) - projection_mat[(2, 3)] / 2.0) * 2.0;
        let z_far =
            -(projection_mat[(2, 3)] / (near_far_ratio * 2.0)) - projection_mat[(2, 3)] / 2.0;
        let z_diff = z_far - z_near;

        let mut spass_config = ShadowMapResult {
            num_splits: splits.len() as u32,
            split_distances: [na::Vector4::default(); 16],
        };

        let spass_config_size: u64 = ShadowMapResult::SHADER_SIZE.into();

        for (i, split) in splits.iter().enumerate() {
            spass_config.split_distances[i].x = z_near + z_diff * split;
        }

        let mut spass_config_contents =
            UniformBuffer::new(Vec::with_capacity(spass_config_size as usize));
        spass_config_contents.write(&spass_config)?;

        use wgpu::util::DeviceExt;
        let spass_config_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: spass_config_contents.into_inner().as_slice(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let depth_tex_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::ClampToBorder,
            address_mode_v: wgpu::AddressMode::ClampToBorder,
            address_mode_w: wgpu::AddressMode::ClampToBorder,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            border_color: Some(wgpu::SamplerBorderColor::OpaqueWhite),
            ..Default::default()
        });

        let mat4_size: u64 = na::Matrix4::<f32>::SHADER_SIZE.into();

        let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: mat4_size * SPLIT_COUNT as u64 * 2,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let out_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &out_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(out_buf.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&depth_tex_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &depth_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(
                        spass_config_buf.as_entire_buffer_binding(),
                    ),
                },
            ],
        });

        Ok(Self {
            splits,
            pnuv_pipeline,
            pipeline,
            bg,
            proj_mat_buf,
            view_mat_buf,
            depth_tex: depth_texture,
            out_bg,
            out_bgl,
            out_buf,
        })
    }

    pub fn out_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.out_bgl
    }

    fn calculate_proj_view_mats(
        light: &PhongLight,
        frustum: &[na::Point3<f32>],
    ) -> (na::Matrix4<f32>, na::Matrix4<f32>) {
        let near_plane_center = frustum[0] + ((frustum[3] - frustum[0]) / 2.0);
        let far_plane_center = frustum[4] + ((frustum[7] - frustum[4]) / 2.0);

        let frustum_center = near_plane_center + (far_plane_center - near_plane_center) / 2.0;

        let radius = (frustum[7] - frustum[0]).norm() / 2.0;

        let tex_per_unit = SHADOW_MAP_SIZE as f32 / (radius * 2.0);
        let scaling = na::Matrix4::new_scaling(tex_per_unit);

        let smap_cam_nonadjusted = na::Matrix4::look_at_rh(
            &na::Point3::new(-light.direction.x, -light.direction.y, -light.direction.z),
            &na::Point3::new(0.0, 0.0, 0.0),
            &na::Vector3::y(),
        ) * scaling;

        let smap_cam_nonadjusted_inv = smap_cam_nonadjusted.try_inverse().unwrap();

        let mut frustum_center_light = smap_cam_nonadjusted.transform_point(&frustum_center);
        frustum_center_light.x = frustum_center_light.x.floor();
        frustum_center_light.y = frustum_center_light.y.floor();
        frustum_center_light = smap_cam_nonadjusted_inv.transform_point(&frustum_center_light);

        let smap_cam_mat = na::Matrix4::look_at_rh(
            &(frustum_center_light - light.direction.xyz()),
            &frustum_center_light,
            &na::Vector3::y(),
        );

        let smap_proj_mat = wgpu_projection(na::Matrix4::new_orthographic(
            -radius, radius, -radius, radius, -radius, radius,
        ));

        (smap_cam_mat, smap_proj_mat)
    }

    pub fn render(
        &self,
        gpu: &Gpu,
        light: &PhongLight,
        camera: &GpuCamera,
        projection_mat: &na::Matrix4<f32>,
        scene: &GpuScene,
    ) -> Result<&wgpu::BindGroup> {
        let full_frustum = calculate_frustum(&camera.look_at_matrix(), projection_mat)?;

        let frustum_splits = split_frustum(&full_frustum, &self.splits);

        let mat4_size: u64 = na::Matrix4::<f32>::SHADER_SIZE.into();
        let offset = mat4_size.max(MIN_UNIFORM_BUFFER_OFFSET_ALIGNMENT);

        for (i, frustum) in frustum_splits.iter().enumerate() {
            let (smap_cam_mat, smap_proj_mat) = Self::calculate_proj_view_mats(light, frustum);

            gpu.queue.write_buffer(
                &self.view_mat_buf,
                i as u64 * offset,
                bytemuck::cast_slice(smap_cam_mat.as_slice()),
            );

            gpu.queue.write_buffer(
                &self.proj_mat_buf,
                i as u64 * offset,
                bytemuck::cast_slice(smap_proj_mat.as_slice()),
            );

            gpu.queue.write_buffer(
                &self.out_buf,
                (i as u64) * mat4_size,
                bytemuck::cast_slice(smap_cam_mat.as_slice()),
            );

            gpu.queue.write_buffer(
                &self.out_buf,
                (i as u64 + SPLIT_COUNT as u64) * mat4_size,
                bytemuck::cast_slice(smap_proj_mat.as_slice()),
            );

            let depth_view = self.depth_tex.create_view(&wgpu::TextureViewDescriptor {
                base_array_layer: i as u32,
                array_layer_count: Some(1),
                dimension: Some(wgpu::TextureViewDimension::D2),
                ..Default::default()
            });

            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                rpass.set_bind_group(
                    0,
                    &self.bg,
                    &[(i as u64 * offset) as u32, (i as u64 * offset) as u32],
                );

                for draw_call in scene.draw_calls() {
                    match draw_call.vertex_array_type {
                        MeshVertexArrayType::PN => {
                            rpass.set_pipeline(&self.pipeline);
                        }
                        MeshVertexArrayType::PNUV => {
                            rpass.set_pipeline(&self.pnuv_pipeline);
                        }
                    }

                    rpass.set_vertex_buffer(
                        0,
                        scene
                            .vertex_buffer_by_type(draw_call.vertex_array_type)
                            .slice(..),
                    );
                    rpass.set_vertex_buffer(
                        1,
                        scene
                            .instance_buffer_by_type(draw_call.instance_type)
                            .slice(..),
                    );

                    if draw_call.indexed {
                        rpass.set_index_buffer(
                            scene.index_buffer().slice(..),
                            wgpu::IndexFormat::Uint32,
                        );

                        rpass.draw_indexed_indirect(
                            scene.indexed_draw_buffer(),
                            draw_call.draw_buffer_offset,
                        );
                    } else {
                        rpass.draw_indirect(
                            scene.non_indexed_draw_buffer(),
                            draw_call.draw_buffer_offset,
                        );
                    }
                }
            }

            gpu.queue.submit(Some(encoder.finish()));
        }

        Ok(&self.out_bg)
    }
}
