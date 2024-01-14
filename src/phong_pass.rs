use crate::{
    camera::GpuCamera, gpu::Gpu, model::GpuModel, projection::GpuProjection,
    world_model::GpuWorldModel,
};
use anyhow::Result;
use encase::{ArrayLength, ShaderType, StorageBuffer};
use nalgebra as na;

pub struct PhongPass {
    bg_layout: wgpu::BindGroupLayout,
    bg: wgpu::BindGroup,
    light_buf: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
    pipeline_layout: wgpu::PipelineLayout,
    shader: wgpu::ShaderModule,
}

#[derive(ShaderType, Clone, Copy)]
pub struct Light {
    light_type: u32,
    position_direction: na::Vector3<f32>,
    color: na::Vector3<f32>,
    angle: f32,
    casting_shadows: u32,
}

impl Light {
    pub fn new_point(position: na::Vector3<f32>, color: na::Vector3<f32>) -> Self {
        Self {
            light_type: 0,
            position_direction: position,
            color,
            angle: 0.0,
            casting_shadows: 0,
        }
    }

    pub fn new_directional(direction: na::Vector3<f32>, color: na::Vector3<f32>) -> Self {
        Self {
            light_type: 1,
            position_direction: direction,
            color,
            angle: 0.0,
            casting_shadows: 0,
        }
    }

    pub fn new_spot(
        position: na::Vector3<f32>,
        direction: na::Vector3<f32>,
        color: na::Vector3<f32>,
        angle: f32,
    ) -> Self {
        Self {
            light_type: 2,
            position_direction: position,
            color,
            angle,
            casting_shadows: 0,
        }
    }

    pub fn toggle_shadow_casting(&mut self) {
        self.casting_shadows = 1;
    }
}

#[derive(ShaderType)]
struct GpuLights {
    size: ArrayLength,
    #[size(runtime)]
    lights: Vec<Light>,
}

impl PhongPass {
    pub fn new(
        gpu: &Gpu,
        camera: &GpuCamera,
        projection: &GpuProjection,
        lights: &[Light],
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
                        visibility: wgpu::ShaderStages::VERTEX,
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

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bg_layout],
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
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

        Ok(Self {
            bg_layout,
            bg,
            light_buf,
            pipeline,
            pipeline_layout,
            shader,
        })
    }

    pub fn render(&self, gpu: &Gpu, world_models: &[&GpuWorldModel]) {
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
            rpass.set_bind_group(0, &self.bg, &[]);

            for model in world_models {
                model.draw(&mut rpass);
            }
        }

        gpu.queue.submit(Some(encoder.finish()));
        frame.present();
    }
}
