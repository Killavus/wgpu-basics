use anyhow::Result;
use encase::{ShaderSize, ShaderType, UniformBuffer};
use nalgebra as na;

use crate::{gpu::Gpu, scene_uniform::SceneUniform, shader_compiler::ShaderCompiler};

use super::geometry_pass::GBuffers;

pub struct SsaoPass {
    ssao_bgl: wgpu::BindGroupLayout,
    samples_buf: wgpu::Buffer,
    output_tex: wgpu::Texture,
    g_sampler: wgpu::Sampler,
    ssao_pipeline: wgpu::RenderPipeline,
}
pub struct SsaoSettings {
    num_samples: u32,
}

fn generate_samples(num_samples: u32) -> Vec<na::Vector3<f32>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut result = Vec::with_capacity(num_samples as usize);

    for i in 0..num_samples {
        // Generate more and more spread samples.
        let factor = (i + 1) as f32 / num_samples as f32;
        let scale = 0.1 + factor * (1.0 - 0.1);

        let mut sample = na::Vector3::new(
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(0.0..1.0),
        );
        sample *= scale;

        result.push(sample);
    }

    result
}

impl SsaoPass {
    pub fn new(
        gpu: &Gpu,
        shader_compiler: &ShaderCompiler,
        ssao_settings: SsaoSettings,
        scene_uniform: &SceneUniform,
    ) -> Result<Self> {
        use wgpu::util::DeviceExt;

        let samples = generate_samples(ssao_settings.num_samples);
        let samples_gpu_size: u64 = samples.size().into();

        let mut samples_contents =
            UniformBuffer::new(Vec::with_capacity(samples_gpu_size as usize));
        samples_contents.write(&samples)?;

        let samples_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SsaoPass::SamplesBuffer"),
                contents: samples_contents.into_inner().as_slice(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let g_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SsaoPass::GSampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let output_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SsaoPass::OutputTexture"),
            size: gpu.viewport_size(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let ssao_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SsaoPass::SsaoBindGroupLayout"),
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
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("SsaoPass::PipelineLayout"),
                bind_group_layouts: &[scene_uniform.layout(), &ssao_bgl],
                push_constant_ranges: &[],
            });

        let module = shader_compiler
            .compilation_unit("./shaders/deferred/ssao.wgsl")?
            .with_integer_def("SSAO_SAMPLES_CNT", ssao_settings.num_samples)
            .compile(&[])?;

        let ssao_shader = gpu.shader_from_module(module);

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("SsaoPass::RenderPipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &ssao_shader,
                    entry_point: "vs_main",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &ssao_shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R8Unorm,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::RED,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

        Ok(Self {
            ssao_bgl,
            output_tex,
            samples_buf,
            g_sampler,
            ssao_pipeline: pipeline,
        })
    }

    pub fn render(
        &self,
        gpu: &Gpu,
        g_buffers: &GBuffers,
        scene_uniform: &SceneUniform,
    ) -> wgpu::TextureView {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let output_tv = self
            .output_tex
            .create_view(&wgpu::TextureViewDescriptor::default());
        let g_normal = g_buffers.g_normal.create_view(&Default::default());

        let depth_tv = gpu.depth_texture_view();

        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SsaoPass::BindGroup"),
            layout: &self.ssao_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(
                        self.samples_buf.as_entire_buffer_binding(),
                    ),
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
                    resource: wgpu::BindingResource::TextureView(&depth_tv),
                },
            ],
        });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SsaoPass::RenderPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_tv,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&self.ssao_pipeline);
            rpass.set_bind_group(0, scene_uniform.bind_group(), &[]);
            rpass.set_bind_group(1, &bg, &[]);
            rpass.draw(0..4, 0..1);
        }

        gpu.queue.submit(Some(encoder.finish()));
        output_tv
    }
}
