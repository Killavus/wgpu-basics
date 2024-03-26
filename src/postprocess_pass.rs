use std::sync::Arc;

use crate::{
    compute::BlurPass, gpu::Gpu, render_context::RenderContext, shader_compiler::ShaderCompiler,
};
use anyhow::Result;
use encase::{ShaderSize, ShaderType, UniformBuffer};
use nalgebra as na;

pub struct PostprocessPass<'window> {
    render_ctx: Arc<RenderContext<'window>>,
    forward_bg: wgpu::BindGroup,
    deferred_bg: wgpu::BindGroup,
    bgl: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    settings_buf: wgpu::Buffer,
    sampler: wgpu::Sampler,
    texture: wgpu::Texture,
}

#[derive(ShaderType, PartialEq)]
pub struct PostprocessSettings {
    bcsg: na::Vector4<f32>,
}

impl PostprocessSettings {
    pub fn saturation_mut(&mut self) -> &mut f32 {
        &mut self.bcsg.z
    }

    pub fn brightness_mut(&mut self) -> &mut f32 {
        &mut self.bcsg.x
    }

    pub fn contrast_mut(&mut self) -> &mut f32 {
        &mut self.bcsg.y
    }

    pub fn gamma_mut(&mut self) -> &mut f32 {
        &mut self.bcsg.w
    }
}

impl Default for PostprocessSettings {
    fn default() -> Self {
        Self::new(0.0, 1.0, 1.0, 0.45)
    }
}

impl PostprocessSettings {
    pub fn new(brightness: f32, contrast: f32, saturation: f32, gamma: f32) -> Self {
        Self {
            bcsg: na::Vector4::new(brightness, contrast, saturation, gamma),
        }
    }
}

impl<'window> PostprocessPass<'window> {
    pub fn new(
        render_ctx: Arc<RenderContext<'window>>,
        deferred_texture: &wgpu::TextureView,
        settings: &PostprocessSettings,
    ) -> Result<Self> {
        let RenderContext {
            gpu,
            shader_compiler,
            ..
        } = render_ctx.as_ref();

        let tex_size = gpu.viewport_size();

        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: tex_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: gpu.swapchain_format(),
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let bgl: wgpu::BindGroupLayout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
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

        let vec4_size: u64 = na::Vector4::<f32>::SHADER_SIZE.into();
        let mut settings_contents = UniformBuffer::new(Vec::with_capacity(vec4_size as usize));
        settings_contents.write(&settings)?;

        use wgpu::util::DeviceExt;
        let settings_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: settings_contents.into_inner().as_slice(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let forward_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(
                        settings_buf.as_entire_buffer_binding(),
                    ),
                },
            ],
        });

        let deferred_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(deferred_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(
                        settings_buf.as_entire_buffer_binding(),
                    ),
                },
            ],
        });

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

        let module = shader_compiler.compilation_unit("./shaders/screenspace/postprocess.wgsl")?;
        let shader = gpu.shader_from_module(module.compile(Default::default())?);

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(gpu.swapchain_format().into())],
                }),
                layout: Some(&pipeline_layout),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

        Ok(Self {
            render_ctx,
            sampler,
            bgl,
            forward_bg,
            deferred_bg,
            pipeline,
            settings_buf,
            texture,
        })
    }

    pub fn on_resize(&mut self, gpu: &Gpu, new_size: (u32, u32)) {
        let tex_size = wgpu::Extent3d {
            width: new_size.0,
            height: new_size.1,
            depth_or_array_layers: 1,
        };

        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: tex_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: gpu.swapchain_format(),
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(
                        self.settings_buf.as_entire_buffer_binding(),
                    ),
                },
            ],
        });

        self.texture = texture;
        self.forward_bg = bg;
    }

    pub fn render(
        &self,
        settings: &PostprocessSettings,
        frame: wgpu::SurfaceTexture,
        deferred: bool,
    ) -> wgpu::SurfaceTexture {
        let RenderContext { gpu, .. } = self.render_ctx.as_ref();

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let settings_size: u64 = PostprocessSettings::SHADER_SIZE.into();
        let mut contents = UniformBuffer::new(Vec::with_capacity(settings_size as usize));

        contents.write(settings).unwrap();

        gpu.queue
            .write_buffer(&self.settings_buf, 0, contents.into_inner().as_slice());

        if !deferred {
            encoder.copy_texture_to_texture(
                frame.texture.as_image_copy(),
                self.texture.as_image_copy(),
                gpu.viewport_size(),
            );
        }

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

            rpass.set_pipeline(&self.pipeline);
            if deferred {
                rpass.set_bind_group(0, &self.deferred_bg, &[]);
            } else {
                rpass.set_bind_group(0, &self.forward_bg, &[]);
            }

            rpass.draw(0..4, 0..1);
        }
        gpu.queue.submit(Some(encoder.finish()));

        frame
    }
}
