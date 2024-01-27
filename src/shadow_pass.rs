use std::num::{NonZeroU32, NonZeroU64};

use anyhow::Result;
use encase::ShaderSize;
use nalgebra as na;

use crate::{
    camera::GpuCamera,
    gpu::Gpu,
    light::{Light, LIGHT_TYPE_DIRECTIONAL},
    model::GpuModel,
    projection::wgpu_projection,
    world_model::GpuWorldModel,
};

pub struct DirectionalShadowPass {
    splits: Vec<f32>,
    pipeline: wgpu::RenderPipeline,
    bg: wgpu::BindGroup,
    depth_tex: wgpu::Texture,
    proj_mat_buf: wgpu::Buffer,
    view_mat_buf: wgpu::Buffer,
    out_bg: wgpu::BindGroup,
    out_bgl: wgpu::BindGroupLayout,
}

const MIN_UNIFORM_BUFFER_OFFSET_ALIGNMENT: u64 = 256;

const SHADOW_MAP_SIZE: (u32, u32) = (2048, 2048);

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
    pub fn new(gpu: &Gpu, splits: Vec<f32>) -> Result<Self> {
        let depth_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE.0,
                height: SHADOW_MAP_SIZE.1,
                depth_or_array_layers: splits.len() as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let shader = gpu.shader_from_file("./shaders/shadowMap.wgsl")?;
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

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipelinel),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[GpuModel::vertex_layout(), GpuWorldModel::instance_layout()],
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
                        count: NonZeroU32::new(splits.len() as u32),
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: NonZeroU32::new(splits.len() as u32),
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
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

        let out_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &out_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(
                        view_mat_buf.as_entire_buffer_binding(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(
                        proj_mat_buf.as_entire_buffer_binding(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&depth_tex_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        &depth_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
            ],
        });

        Ok(Self {
            splits,
            pipeline,
            bg,
            proj_mat_buf,
            view_mat_buf,
            depth_tex: depth_texture,
            out_bg,
            out_bgl,
        })
    }

    pub fn out_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.out_bgl
    }

    fn calculate_proj_view_mats(
        light: &Light,
        frustum: &[na::Point3<f32>],
    ) -> (na::Matrix4<f32>, na::Matrix4<f32>) {
        let near_plane_center = frustum[0] + ((frustum[3] - frustum[0]) / 2.0);
        let far_plane_center = frustum[4] + ((frustum[7] - frustum[4]) / 2.0);

        let frustum_center = near_plane_center + (far_plane_center - near_plane_center) / 2.0;

        let radius = ((frustum[7] - frustum[0]).norm() / 2.0);

        let tex_per_unit = SHADOW_MAP_SIZE.0 as f32 / (radius * 2.0);
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
            &(frustum_center_light - light.direction),
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
        light: &Light,
        camera: &GpuCamera,
        projection_mat: &na::Matrix4<f32>,
        world_models: &[&GpuWorldModel],
    ) -> Result<&wgpu::BindGroup> {
        if light.light_type != LIGHT_TYPE_DIRECTIONAL {
            return Err(anyhow::anyhow!("light type is not directional"));
        }

        let full_frustum =
            calculate_frustum(&camera.look_at_matrix(), &wgpu_projection(*projection_mat))?;

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

            let depth_view = self.depth_tex.create_view(&wgpu::TextureViewDescriptor {
                base_array_layer: i as u32,
                array_layer_count: Some(1),
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

                rpass.set_pipeline(&self.pipeline);
                rpass.set_bind_group(
                    0,
                    &self.bg,
                    &[(i as u64 * offset) as u32, (i as u64 * offset) as u32],
                );

                for objects in world_models {
                    objects.draw(&mut rpass);
                }
            }

            gpu.queue.submit(Some(encoder.finish()));
        }

        Ok(&self.out_bg)
    }
}