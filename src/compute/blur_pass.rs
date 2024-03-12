use anyhow::Result;

use crate::{gpu::Gpu, shader_compiler::ShaderCompiler};

pub struct BlurPass {
    compute_pipeline: wgpu::ComputePipeline,
    blur_tex_a: wgpu::Texture,
    blur_tex_b: wgpu::Texture,
    bg_ax: wgpu::BindGroup,
    bg_ay: wgpu::BindGroup,
    bg_bx: wgpu::BindGroup,
    bg_by: wgpu::BindGroup,
    filter_size_buf: wgpu::Buffer,
}

impl BlurPass {
    pub fn new(
        gpu: &Gpu,
        shader_compiler: &ShaderCompiler,
        input_size: wgpu::Extent3d,
        input_format: wgpu::TextureFormat,
    ) -> Result<Self> {
        let blur_tex_a = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("BlurPass::TextureA"),
            size: input_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: input_format,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let blur_tex_b = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("BlurPass::TextureB"),
            size: input_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: input_format,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
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

        let shader = gpu.shader_from_module(
            shader_compiler
                .compilation_unit("./shaders/compute/blur.wgsl")?
                .compile(&[])?,
        );

        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("BlurPass::Sampler"),
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
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
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

        let blur_a_tv = blur_tex_a.create_view(&Default::default());
        let blur_b_tv = blur_tex_b.create_view(&Default::default());

        let bg_ax = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BlurPass::BindGroup"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&blur_b_tv),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&blur_a_tv),
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

        let bg_ay = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BlurPass::BindGroup"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&blur_b_tv),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&blur_a_tv),
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

        let bg_bx = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BlurPass::BindGroup"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&blur_a_tv),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&blur_b_tv),
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

        let bg_by = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BlurPass::BindGroup"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&blur_a_tv),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&blur_b_tv),
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
            blur_tex_a,
            blur_tex_b,
            bg_ax,
            bg_ay,
            bg_bx,
            bg_by,
            filter_size_buf,
        })
    }

    pub fn perform(
        &self,
        gpu: &Gpu,
        input: &wgpu::Texture,
        iterations: u32,
        filter_size: u32,
    ) -> wgpu::TextureView {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("BlurPass::CommandEncoder"),
            });

        encoder.copy_texture_to_texture(
            input.as_image_copy(),
            self.blur_tex_a.as_image_copy(),
            input.size(),
        );

        gpu.queue.write_buffer(
            &self.filter_size_buf,
            0,
            bytemuck::cast_slice(&[filter_size]),
        );
        let wgpu::Extent3d {
            width: image_width,
            height: image_height,
            ..
        } = self.blur_tex_a.size();

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BlurPass::ComputePass"),
                timestamp_writes: None,
            });

            cpass.set_pipeline(&self.compute_pipeline);

            for i in 0..iterations {
                let input_select = i % 2;

                let (bg_x, bg_y) = if input_select == 0 {
                    (&self.bg_ax, &self.bg_ay)
                } else {
                    (&self.bg_bx, &self.bg_by)
                };

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

        if iterations % 2 == 0 {
            self.blur_tex_a.create_view(&Default::default())
        } else {
            self.blur_tex_b.create_view(&Default::default())
        }
    }
}
