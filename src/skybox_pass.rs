use crate::{
    camera::GpuCamera,
    gpu::Gpu,
    model::{Cube, GpuModel},
    projection::GpuProjection,
    scene_uniform::SceneUniform,
    world_model::{GpuWorldModel, WorldModel},
};
use anyhow::Result;
use nalgebra as na;

pub struct SkyboxPass {
    bg: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    cube_model: GpuWorldModel,
    skybox_tex: wgpu::Texture,
    skybox_sampler: wgpu::Sampler,
}

// Idea is like this:
// 1. Render the scene (phong pass)
// 2. Take camera, remove its translation.
// 3. Render the skybox with depth test inverted.

impl SkyboxPass {
    pub fn new(
        gpu: &Gpu,
        scene_uniform: &SceneUniform,
        skybox_tex: wgpu::Texture,
        skybox_sampler: wgpu::Sampler,
    ) -> Result<Self> {
        let mut cube_model = WorldModel::new(Cube::new().model());
        cube_model.add(na::Matrix4::identity(), na::Vector3::zeros());
        let cube_model = cube_model.into_gpu(&gpu.device);

        let tex_view = skybox_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
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
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&skybox_sampler),
                },
            ],
        });

        let shader = gpu.shader_from_file("./shaders/skybox.wgsl")?;

        let pipelinel = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[scene_uniform.layout(), &bgl],
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
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(gpu.swapchain_format().into())],
                }),
                multiview: None,
            });

        Ok(Self {
            bg,
            pipeline,
            bgl,
            cube_model,
            skybox_tex,
            skybox_sampler,
        })
    }

    pub fn render(
        &self,
        gpu: &Gpu,
        scene_uniform: &SceneUniform,
        frame: wgpu::SurfaceTexture,
    ) -> wgpu::SurfaceTexture {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

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
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, scene_uniform.bind_group(), &[]);
            rpass.set_bind_group(1, &self.bg, &[]);

            self.cube_model.draw(&mut rpass);
        }

        gpu.queue.submit(Some(encoder.finish()));
        frame
    }
}
