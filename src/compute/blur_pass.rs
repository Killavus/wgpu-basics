use anyhow::Result;

use crate::{gpu::Gpu, shader_compiler::ShaderCompiler};

pub struct BlurPass {
    compute_pipeline: wgpu::ComputePipeline,
    blur_tex_x: wgpu::Texture,
    bg_x: wgpu::BindGroup,
    bg_y: wgpu::BindGroup,
    flip_x: wgpu::Buffer,
    sampler: wgpu::Sampler,
    filter_size_buf: wgpu::Buffer,
}

impl BlurPass {
    pub fn new(
        gpu: &Gpu,
        shader_compiler: &ShaderCompiler,
        input_size: wgpu::Extent3d,
        input_format: wgpu::TextureFormat,
    ) -> Result<Self> {
        let blur_tex_x = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("BlurPass::TextureX"),
            size: input_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: input_format,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let blur_tex_y = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("BlurPass::TextureY"),
            size: input_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: input_format,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        use wgpu::util::DeviceExt;
        let flip_x_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("BlurPass::FlipXBuffer"),
                contents: bytemuck::cast_slice(&[0u32]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let flip_y_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("BlurPass::FlipYBuffer"),
                contents: bytemuck::cast_slice(&[1u32]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let filter_size_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BlurPass::FilterSizeBuffer"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let variant = match input_format {
            wgpu::TextureFormat::Rgba8Unorm => "RGBA8UNORM",
            wgpu::TextureFormat::Rgba16Float => "RGBA16FLOAT",
            wgpu::TextureFormat::R8Unorm => "R8UNORM",
            wgpu::TextureFormat::Bgra8Unorm => "BGRA8UNORM",
            _ => "RGBA8UNORM",
        };

        let shader = gpu.shader_from_module(
            shader_compiler
                .compilation_unit("./shaders/compute/blur.wgsl")?
                .compile(&[variant])?,
        );

        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("BlurPass::Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("BlurPass::BindGroupLayout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: input_format,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let blur_x_tv = blur_tex_x.create_view(&Default::default());
        let blur_y_tv = blur_tex_y.create_view(&Default::default());

        let bg_y = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BlurPass::BindGroupY"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    // dst
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&blur_y_tv),
                },
                wgpu::BindGroupEntry {
                    // src
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&blur_x_tv),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(flip_y_buf.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(
                        filter_size_buf.as_entire_buffer_binding(),
                    ),
                },
            ],
        });

        let bg_x = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BlurPass::BindGroupX"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&blur_x_tv),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&blur_y_tv),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(flip_x_buf.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(
                        filter_size_buf.as_entire_buffer_binding(),
                    ),
                },
            ],
        });

        let compute_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("BlurPass::PipelineLayout"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

        let compute_pipeline =
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("BlurPass::Pipeline"),
                    layout: Some(&compute_layout),
                    module: &shader,
                    entry_point: "blur",
                });

        Ok(Self {
            compute_pipeline,
            flip_x: flip_x_buf,
            blur_tex_x,
            bg_x,
            sampler,
            bg_y,
            filter_size_buf,
        })
    }

    pub fn perform(
        &self,
        gpu: &Gpu,
        input: &wgpu::Texture,
        iterations: u32,
        filter_size: u32,
    ) -> &wgpu::Texture {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("BlurPass::CommandEncoder"),
            });

        gpu.queue.write_buffer(
            &self.filter_size_buf,
            0,
            bytemuck::cast_slice(&[filter_size]),
        );
        let wgpu::Extent3d {
            width: image_width,
            height: image_height,
            ..
        } = self.blur_tex_x.size();

        let source_tv = input.create_view(&Default::default());
        let tex_x_tv = self.blur_tex_x.create_view(&Default::default());
        let bg_source = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&tex_x_tv),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&source_tv),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(self.flip_x.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(
                        self.filter_size_buf.as_entire_buffer_binding(),
                    ),
                },
            ],
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BlurPass::ComputePass"),
                timestamp_writes: None,
            });

            cpass.set_pipeline(&self.compute_pipeline);

            cpass.set_bind_group(0, &bg_source, &[]);
            cpass.dispatch_workgroups(
                ((image_width as f64) / (128 - filter_size - 1) as f64).ceil() as u32,
                (image_height as f32 / 4.0).ceil() as u32,
                1,
            );

            cpass.set_bind_group(0, &self.bg_y, &[]);
            cpass.dispatch_workgroups(
                ((image_height as f64) / (128 - filter_size - 1) as f64).ceil() as u32,
                (image_width as f32 / 4.0).ceil() as u32,
                1,
            );

            for _ in 0..iterations - 1 {
                let (bg_x, bg_y) = (&self.bg_x, &self.bg_y);

                cpass.set_bind_group(0, bg_x, &[]);
                cpass.dispatch_workgroups(
                    ((image_width as f64) / (128 - filter_size - 1) as f64).ceil() as u32,
                    (image_height as f32 / 4.0).ceil() as u32,
                    1,
                );
                cpass.set_bind_group(0, bg_y, &[]);
                cpass.dispatch_workgroups(
                    ((image_height as f64) / (128 - filter_size - 1) as f64).ceil() as u32,
                    (image_width as f32 / 4.0).ceil() as u32,
                    1,
                );
            }
        }

        gpu.queue.submit(Some(encoder.finish()));
        &self.blur_tex_x
    }
}
