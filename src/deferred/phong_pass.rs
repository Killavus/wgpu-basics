use std::sync::Arc;

use crate::render_context::RenderContext;
use anyhow::Result;
use encase::{ShaderType, StorageBuffer};

use super::geometry_pass::GBuffers;

pub struct PhongPass<'window> {
    render_ctx: Arc<RenderContext<'window>>,
    pipeline: wgpu::RenderPipeline,
    light_buf: wgpu::Buffer,
    g_sampler: wgpu::Sampler,
    output_tex: wgpu::Texture,
    fill_bgl: wgpu::BindGroupLayout,
}

impl<'window> PhongPass<'window> {
    pub fn new(
        render_ctx: Arc<RenderContext<'window>>,
        shadow_bgl: &wgpu::BindGroupLayout,
    ) -> Result<Self> {
        let RenderContext {
            gpu,
            shader_compiler,
            scene_uniform,
            light_scene: lights,
            ..
        } = render_ctx.as_ref();

        let fill_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    // g_Normal
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // g_Diffuse
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // g_Specular
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Depth texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Ssao tex
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let gpu_lights = lights.into_gpu();
        let gpu_lights_size: u64 = gpu_lights.size().into();
        let mut light_contents = StorageBuffer::new(Vec::with_capacity(gpu_lights_size as usize));
        light_contents.write(&gpu_lights)?;

        let output = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: gpu.viewport_size(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        use wgpu::util::DeviceExt;

        let light_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: light_contents.into_inner().as_slice(),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let g_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let module = shader_compiler
            .compilation_unit("./shaders/deferred/phong.wgsl")?
            .with_def("DEFERRED")
            .with_def("SHADOW_MAP")
            .compile(&[])?;

        let fill_shader = gpu.shader_from_module(module);

        let fill_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[scene_uniform.layout(), &fill_bgl, shadow_bgl],
                    push_constant_ranges: &[],
                });

        let fill_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&fill_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &fill_shader,
                    entry_point: "vs_main",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fill_shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                depth_stencil: None,
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

        Ok(Self {
            render_ctx,
            fill_bgl,
            light_buf,
            g_sampler,
            pipeline: fill_pipeline,
            output_tex: output,
        })
    }

    pub fn output_tex_view(&self) -> wgpu::TextureView {
        self.output_tex.create_view(&Default::default())
    }

    pub fn render(
        &self,
        g_buffers: &GBuffers,
        spass_bg: &wgpu::BindGroup,
        ssao_tex: &wgpu::TextureView,
    ) {
        let RenderContext {
            gpu, scene_uniform, ..
        } = self.render_ctx.as_ref();

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let (g_normal, g_diffuse, g_specular) = (
            g_buffers.g_normal.create_view(&Default::default()),
            g_buffers.g_diffuse.create_view(&Default::default()),
            g_buffers.g_specular.create_view(&Default::default()),
        );

        let fill_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.fill_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.light_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.g_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&g_normal),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&g_diffuse),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&g_specular),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&gpu.depth_texture_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(ssao_tex),
                },
            ],
        });

        let output_tv = self.output_tex.create_view(&Default::default());

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_tv,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, scene_uniform.bind_group(), &[]);
            rpass.set_bind_group(1, &fill_bg, &[]);
            rpass.set_bind_group(2, spass_bg, &[]);

            rpass.draw(0..4, 0..1);
        }

        gpu.queue.submit(Some(encoder.finish()));
    }
}
