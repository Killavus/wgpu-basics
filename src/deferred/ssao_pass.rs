use anyhow::Result;
use encase::{ShaderType, UniformBuffer};
use nalgebra as na;
use rand::distributions::Uniform;

use crate::{
    compute::BlurPass, gpu::Gpu, scene_uniform::SceneUniform, shader_compiler::ShaderCompiler,
};

use super::geometry_pass::GBuffers;

pub struct SsaoPass {
    ssao_bgl: wgpu::BindGroupLayout,
    samples_buf: wgpu::Buffer,
    output_tex: wgpu::Texture,
    g_sampler: wgpu::Sampler,
    noise_sampler: wgpu::Sampler,
    noise_tex: wgpu::Texture,
    ssao_pipeline: wgpu::RenderPipeline,
    blur_pass: BlurPass,
}

const NUM_SAMPLES: usize = 64;
const NOISE_TEX_SIZE: usize = 16;
const NOISE_TEX_DIM: usize = 4;

fn generate_samples() -> [na::Vector3<f32>; NUM_SAMPLES] {
    use rand::distributions::Distribution;
    let mut rng = rand::thread_rng();

    let mut result = [na::Vector3::zeros(); NUM_SAMPLES];

    for (i, sample) in result.iter_mut().enumerate() {
        // Generate more and more spread samples.
        let factor = (i + 1) as f32 / NUM_SAMPLES as f32;
        let scale = 0.1 + factor * (1.0 - 0.1);

        let distribution = Uniform::new(-1.0, 1.0);

        *sample = na::Vector3::new(
            distribution.sample(&mut rng),
            distribution.sample(&mut rng),
            distribution.sample(&mut rng) * 0.5 + 0.5,
        );
        *sample *= distribution.sample(&mut rng) * 0.5 + 0.5;
        *sample *= scale;
    }

    result
}

fn generate_noise() -> [na::Vector4<f32>; NOISE_TEX_SIZE] {
    use rand::distributions::Distribution;
    let mut rng = rand::thread_rng();

    let mut result = [na::Vector4::zeros(); NOISE_TEX_SIZE];
    let distribution = Uniform::new(-1.0, 1.0);

    for sample in result.iter_mut() {
        *sample = na::Vector4::new(
            distribution.sample(&mut rng),
            distribution.sample(&mut rng),
            0.0,
            0.0,
        )
        .normalize();
    }

    result
}

impl SsaoPass {
    pub fn new(
        gpu: &Gpu,
        shader_compiler: &ShaderCompiler,
        scene_uniform: &SceneUniform,
    ) -> Result<Self> {
        use wgpu::util::DeviceExt;

        let samples = generate_samples();
        let samples_gpu_size: u64 = samples.size().into();

        let noise = generate_noise();
        let noise_flat = noise
            .iter()
            .flat_map(|v| v.as_slice().iter().copied())
            .collect::<Vec<_>>();

        let noise_tex = gpu.device.create_texture_with_data(
            &gpu.queue,
            &wgpu::TextureDescriptor {
                label: Some("SsaoPass::NoiseTexture"),
                size: wgpu::Extent3d {
                    width: NOISE_TEX_DIM as u32,
                    height: NOISE_TEX_DIM as u32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(noise_flat.as_slice()),
        );

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

        let noise_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SsaoPass::GSampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
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
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
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
            .with_integer_def("SSAO_SAMPLES_CNT", NUM_SAMPLES as u32)
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

        let blur_pass =
            BlurPass::new(gpu, shader_compiler, output_tex.size(), output_tex.format())?;

        Ok(Self {
            ssao_bgl,
            output_tex,
            samples_buf,
            g_sampler,
            noise_sampler,
            noise_tex,
            ssao_pipeline: pipeline,
            blur_pass,
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
        let noise_tv = self.noise_tex.create_view(&Default::default());

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
                    resource: wgpu::BindingResource::Sampler(&self.noise_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&g_normal),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&noise_tv),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
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
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
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

        self.blur_pass
            .perform(gpu, &self.output_tex, 8, 4)
            .create_view(&Default::default())
    }
}
