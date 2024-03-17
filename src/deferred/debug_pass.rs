use crate::{gpu::Gpu, shader_compiler::ShaderCompiler};
use anyhow::Result;

use super::geometry_pass::GBuffers;

#[derive(Default, PartialEq, Eq)]
pub enum DeferredDebug {
    #[default]
    Normals,
    Diffuse,
    Specular,
    Depth,
    AmbientOcclusion,
}

pub struct DebugPass {
    pipeline_depth: wgpu::RenderPipeline,
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
}

impl DebugPass {
    pub fn new(gpu: &Gpu, shader_compiler: &ShaderCompiler) -> Result<Self> {
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });

        let bgl_depth = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Depth,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });

        let module = shader_compiler.compilation_unit("./shaders/showTexture.wgsl")?;
        let shader = gpu.shader_from_module(module.compile(&[])?);
        let depth_shader = gpu.shader_from_module(module.compile(&["DEPTH_TEXTURE"])?);

        let pipeline_l = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

        let pipeline_depth_l = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl_depth],
                push_constant_ranges: &[],
            });

        let [pipeline, pipeline_depth] = [(shader, pipeline_l), (depth_shader, pipeline_depth_l)]
            .map(|(shader, layout)| {
                gpu.device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: None,
                        layout: Some(&layout),
                        vertex: wgpu::VertexState {
                            module: &shader,
                            entry_point: "vs_main",
                            buffers: &[],
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &shader,
                            entry_point: "fs_main",
                            targets: &[Some(wgpu::ColorTargetState {
                                format: gpu.swapchain_format(),
                                blend: Some(wgpu::BlendState::REPLACE),
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                        }),
                        primitive: wgpu::PrimitiveState {
                            topology: wgpu::PrimitiveTopology::TriangleStrip,
                            ..Default::default()
                        },
                        depth_stencil: None,
                        multisample: wgpu::MultisampleState::default(),
                        multiview: None,
                    })
            });

        Ok(Self {
            pipeline_depth,
            pipeline,
            sampler,
        })
    }

    pub fn render(
        &self,
        gpu: &Gpu,
        g_bufs: &GBuffers,
        frame: &wgpu::SurfaceTexture,
        debug_type: &DeferredDebug,
    ) {
        let bg = match debug_type {
            DeferredDebug::Normals => {
                let tv = g_bufs
                    .g_normal
                    .create_view(&wgpu::TextureViewDescriptor::default());

                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("DeferredDebug::NormalsBG"),
                    layout: &self.pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&tv),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                })
            }
            DeferredDebug::Diffuse => {
                let tv = g_bufs
                    .g_diffuse
                    .create_view(&wgpu::TextureViewDescriptor::default());

                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("DeferredDebug::DiffuseBG"),
                    layout: &self.pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&tv),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                })
            }
            DeferredDebug::Specular => {
                let tv = g_bufs
                    .g_specular
                    .create_view(&wgpu::TextureViewDescriptor::default());

                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("DeferredDebug::SpecularBG"),
                    layout: &self.pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&tv),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                })
            }
            DeferredDebug::Depth => {
                let tv = gpu.depth_texture_view();

                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("DeferredDebug::DepthBG"),
                    layout: &self.pipeline_depth.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&tv),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                })
            }
            DeferredDebug::AmbientOcclusion => {
                let tv = g_bufs
                    .g_specular
                    .create_view(&wgpu::TextureViewDescriptor::default());

                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("DeferredDebug::AOBG"),
                    layout: &self.pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&tv),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                })
            }
        };

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let frame_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame_view,
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

            match debug_type {
                DeferredDebug::Depth => {
                    rpass.set_pipeline(&self.pipeline_depth);
                }
                _ => {
                    rpass.set_pipeline(&self.pipeline);
                }
            }

            rpass.set_bind_group(0, &bg, &[]);
            rpass.draw(0..4, 0..1);
        }
        gpu.queue.submit(Some(encoder.finish()));
    }
}
