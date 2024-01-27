use crate::{
    camera::GpuCamera,
    gpu::Gpu,
    light::{GpuLights, Light},
    model::GpuModel,
    projection::GpuProjection,
    world_model::GpuWorldModel,
};
use anyhow::Result;
use encase::{ArrayLength, ShaderSize, ShaderType, StorageBuffer, UniformBuffer};

pub struct PhongPass {
    pass_bgl: wgpu::BindGroupLayout,
    pass_bg: wgpu::BindGroup,
    settings_bgl: wgpu::BindGroupLayout,
    settings_bg: wgpu::BindGroup,
    light_buf: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
    pipeline_layout: wgpu::PipelineLayout,
    shader: wgpu::ShaderModule,
    settings: PhongSettings,
    settings_buf: wgpu::Buffer,
}

#[derive(ShaderType)]
pub struct PhongSettings {
    pub ambient_strength: f32,
    pub diffuse_strength: f32,
    pub specular_strength: f32,
    pub specular_coefficient: f32,
}

impl PhongPass {
    pub fn new(
        gpu: &Gpu,
        camera: &GpuCamera,
        projection: &GpuProjection,
        lights: &[Light],
        shadow_bgl: &wgpu::BindGroupLayout,
        settings: PhongSettings,
    ) -> Result<Self> {
        use wgpu::util::DeviceExt;

        let gpu_lights = GpuLights {
            size: ArrayLength,
            lights: lights.to_vec(),
        };

        let gpu_lights_size: u64 = gpu_lights.size().into();
        let mut light_contents = StorageBuffer::new(Vec::with_capacity(gpu_lights_size as usize));
        light_contents.write(&gpu_lights)?;

        let light_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: light_contents.into_inner().as_slice(),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let settings_size: u64 = PhongSettings::SHADER_SIZE.into();
        let mut settings_contents = UniformBuffer::new(Vec::with_capacity(settings_size as usize));
        settings_contents.write(&settings)?;
        let settings_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: settings_contents.into_inner().as_slice(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let shader = gpu.shader_from_file("./shaders/phong.wgsl")?;

        let bg_layout = gpu
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
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: camera.model_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: projection.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: light_buf.as_entire_binding(),
                },
            ],
        });

        let settings_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let settings_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &settings_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: settings_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bg_layout, &settings_bgl, &shadow_bgl],
                push_constant_ranges: &[],
            });

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[GpuModel::vertex_layout(), GpuWorldModel::instance_layout()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(gpu.swapchain_format().into())],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

        Ok(Self {
            pass_bgl: bg_layout,
            pass_bg: bg,
            light_buf,
            pipeline,
            pipeline_layout,
            shader,
            settings,
            settings_buf,
            settings_bgl,
            settings_bg,
        })
    }

    pub fn render(
        &self,
        gpu: &Gpu,
        world_models: &[&GpuWorldModel],
        shadow_bg: &wgpu::BindGroup,
    ) -> wgpu::SurfaceTexture {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let frame = gpu.current_texture();
        {
            let frame_view = frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let depth_view = gpu.depth_texture_view();

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
            rpass.set_bind_group(0, &self.pass_bg, &[]);
            rpass.set_bind_group(1, &self.settings_bg, &[]);
            rpass.set_bind_group(2, shadow_bg, &[]);

            for model in world_models {
                model.draw(&mut rpass);
            }
        }

        gpu.queue.submit(Some(encoder.finish()));
        frame
    }
}
