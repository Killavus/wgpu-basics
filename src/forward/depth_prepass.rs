use crate::{
    gpu::Gpu,
    mesh::{Mesh, MeshVertexArrayType},
    scene::{GpuScene, Instance},
    scene_uniform::SceneUniform,
    shader_compiler::ShaderCompiler,
};
use anyhow::Result;
use naga_oil::compose::ShaderDefValue;

pub struct DepthPrepass {
    pn_pipeline: wgpu::RenderPipeline,
    pnuv_pipeline: wgpu::RenderPipeline,
    pntbuv_pipeline: wgpu::RenderPipeline,
}

impl DepthPrepass {
    pub fn new(
        gpu: &Gpu,
        shader_compiler: &mut ShaderCompiler,
        scene_uniform: &SceneUniform,
    ) -> Result<Self> {
        let shader = gpu.shader_from_module(shader_compiler.compile(
            "./shaders/shadow/csm.wgsl",
            vec![("VERTEX_PN".into(), ShaderDefValue::Bool(true))],
        )?);

        let pnuv_shader = gpu.shader_from_module(shader_compiler.compile(
            "./shaders/shadow/csm.wgsl",
            vec![("VERTEX_PNUV".into(), ShaderDefValue::Bool(true))],
        )?);

        let pntbuv_shader = gpu.shader_from_module(shader_compiler.compile(
            "./shaders/shadow/csm.wgsl",
            vec![("VERTEX_PNTBUV".into(), ShaderDefValue::Bool(true))],
        )?);

        let pipelinel = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[scene_uniform.layout()],
                push_constant_ranges: &[],
            });

        let pn_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipelinel),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[
                        Mesh::pn_vertex_layout(),
                        Instance::pn_model_instance_layout(),
                    ],
                },
                fragment: None,
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

        let pnuv_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipelinel),
                vertex: wgpu::VertexState {
                    module: &pnuv_shader,
                    entry_point: "vs_main",
                    buffers: &[
                        Mesh::pnuv_vertex_layout(),
                        Instance::pnuv_model_instance_layout(),
                    ],
                },
                fragment: None,
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

        let pntbuv_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipelinel),
                vertex: wgpu::VertexState {
                    module: &pntbuv_shader,
                    entry_point: "vs_main",
                    buffers: &[
                        Mesh::pntbuv_vertex_layout(),
                        Instance::pntbuv_model_instance_layout(),
                    ],
                },
                fragment: None,
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
            pn_pipeline,
            pnuv_pipeline,
            pntbuv_pipeline,
        })
    }

    pub fn render(&self, gpu: &Gpu, scene_uniform: &SceneUniform, scene: &GpuScene) {
        let depth_view = gpu.depth_texture_view();
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[],
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

            rpass.set_bind_group(0, scene_uniform.bind_group(), &[]);

            for draw_call in scene.draw_calls() {
                match draw_call.vertex_array_type {
                    MeshVertexArrayType::PNUV => rpass.set_pipeline(&self.pnuv_pipeline),
                    MeshVertexArrayType::PNTBUV => rpass.set_pipeline(&self.pntbuv_pipeline),
                    MeshVertexArrayType::PN => rpass.set_pipeline(&self.pn_pipeline),
                };

                rpass.set_vertex_buffer(
                    0,
                    scene
                        .vertex_buffer_by_type(draw_call.vertex_array_type)
                        .slice(..),
                );
                rpass.set_vertex_buffer(
                    1,
                    scene
                        .instance_buffer_by_type(draw_call.instance_type)
                        .slice(..),
                );

                if draw_call.indexed {
                    rpass.set_index_buffer(
                        scene.index_buffer().slice(..),
                        wgpu::IndexFormat::Uint32,
                    );

                    rpass.draw_indexed_indirect(
                        scene.indexed_draw_buffer(),
                        draw_call.draw_buffer_offset,
                    );
                } else {
                    rpass.draw_indirect(
                        scene.non_indexed_draw_buffer(),
                        draw_call.draw_buffer_offset,
                    );
                }
            }
        }

        gpu.queue.submit(Some(encoder.finish()));
    }
}
