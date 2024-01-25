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
    pipeline: wgpu::RenderPipeline,
    bg: wgpu::BindGroup,
    depth_tex: wgpu::Texture,
    proj_mat_buf: wgpu::Buffer,
    view_mat_buf: wgpu::Buffer,
    out_bg: wgpu::BindGroup,
    out_bgl: wgpu::BindGroupLayout,
}

const SHADOW_MAP_SIZE: (u32, u32) = (1024, 1024);

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

impl DirectionalShadowPass {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let depth_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE.0,
                height: SHADOW_MAP_SIZE.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let shader = gpu.shader_from_file("./shaders/shadowMap.wgsl")?;

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
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
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

        let mat4_size: u64 = na::Matrix4::<f32>::SHADER_SIZE.into();
        let view_mat_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: mat4_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let proj_mat_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: mat4_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
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
            ],
        });

        let out_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
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
                            view_dimension: wgpu::TextureViewDimension::D2,
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

        let frustum =
            calculate_frustum(&camera.look_at_matrix(), &wgpu_projection(*projection_mat))?;

        let near_plane_center = frustum[0] + ((frustum[3] - frustum[0]) / 2.0);
        let far_plane_center = frustum[4] + ((frustum[7] - frustum[4]) / 2.0);

        let frustum_center = near_plane_center + (far_plane_center - near_plane_center) / 2.0;

        //let d = light.direction * (far_plane_center.z - near_plane_center.z);

        let smap_cam_mat: na::Matrix<f32, na::Const<4>, na::Const<4>, na::ArrayStorage<f32, 4, 4>> =
            na::Matrix4::look_at_rh(
                &na::Point3::new(-light.direction.x, -light.direction.y, -light.direction.z),
                &na::Point3::new(0.0, 0.0, 0.0),
                &na::Vector3::y(),
            );

        let (mut xmin, mut xmax, mut ymin, mut ymax, mut zmin, mut zmax) = (
            std::f32::MAX,
            std::f32::MIN,
            std::f32::MAX,
            std::f32::MIN,
            std::f32::MAX,
            std::f32::MIN,
        );

        for p in frustum.iter() {
            let p = smap_cam_mat.transform_point(p);
            xmin = xmin.min(p.x);
            xmax = xmax.max(p.x);
            ymin = ymin.min(p.y);
            ymax = ymax.max(p.y);
            zmin = zmin.min(p.z);
            zmax = zmax.max(p.z);
        }

        // Defined by AABB of the frustum.
        let smap_proj_mat = wgpu_projection(na::Matrix4::new_orthographic(
            xmin, xmax, ymin, ymax, zmin, zmax,
        ));

        gpu.queue.write_buffer(
            &self.view_mat_buf,
            0,
            bytemuck::cast_slice(smap_cam_mat.as_slice()),
        );

        gpu.queue.write_buffer(
            &self.proj_mat_buf,
            0,
            bytemuck::cast_slice(smap_proj_mat.as_slice()),
        );

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let depth_view = self
            .depth_tex
            .create_view(&wgpu::TextureViewDescriptor::default());
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
            rpass.set_bind_group(0, &self.bg, &[]);

            for objects in world_models {
                objects.draw(&mut rpass);
            }
        }

        gpu.queue.submit(Some(encoder.finish()));
        gpu.device.poll(wgpu::Maintain::Wait);
        Ok(&self.out_bg)
    }
}
