use crate::{
    gpu::Gpu,
    mesh::{Mesh, MeshBuilder},
    scene_uniform::SceneUniform,
    shader_compiler::ShaderCompiler,
    shapes::Cube,
};
use anyhow::Result;

pub struct SkyboxPass {
    bg: wgpu::BindGroup,
    rgba8_pipeline: wgpu::RenderPipeline,
    rgba16_pipeline: wgpu::RenderPipeline,
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
}

impl SkyboxPass {
    pub fn new(
        gpu: &Gpu,
        shader_compiler: &mut ShaderCompiler,
        scene_uniform: &SceneUniform,
        skybox_tex: wgpu::Texture,
        skybox_sampler: wgpu::Sampler,
    ) -> Result<Self> {
        let cube_mesh = MeshBuilder::new().with_geometry(Cube::geometry()).build()?;
        let mut cube_vbuf = vec![];
        let mut cube_index = vec![];
        cube_mesh.copy_to_mesh_bank(&mut cube_vbuf);
        cube_mesh.copy_to_index_buffer(&mut cube_index);

        use wgpu::util::DeviceExt;

        let vbuf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: cube_vbuf.as_slice(),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let ibuf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(cube_index.as_slice()),
                usage: wgpu::BufferUsages::INDEX,
            });

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

        let shader = gpu.shader_from_module(
            shader_compiler
                .compilation_unit("./shaders/skybox/simple.wgsl")?
                .compile(&[])?,
        );

        let pipelinel = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[scene_uniform.layout(), &bgl],
                push_constant_ranges: &[],
            });

        let rgba8_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipelinel),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[Mesh::pn_vertex_layout()],
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

        let rgba16_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipelinel),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[Mesh::pn_vertex_layout()],
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
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
            });

        Ok(Self {
            bg,
            rgba8_pipeline,
            rgba16_pipeline,
            vbuf,
            ibuf,
        })
    }

    pub fn render(
        &self,
        gpu: &Gpu,
        scene_uniform: &SceneUniform,
        output_tv: wgpu::TextureView,
        hdr: bool,
    ) {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        {
            let frame_view = output_tv;
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

            if hdr {
                rpass.set_pipeline(&self.rgba16_pipeline);
            } else {
                rpass.set_pipeline(&self.rgba8_pipeline);
            }

            rpass.set_bind_group(0, scene_uniform.bind_group(), &[]);
            rpass.set_bind_group(1, &self.bg, &[]);

            rpass.set_vertex_buffer(0, self.vbuf.slice(..));
            rpass.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..36, 0, 0..1);
        }

        gpu.queue.submit(Some(encoder.finish()));
    }
}
