use anyhow::Result;
use naga_oil::compose::ShaderDefValue;

use crate::{
    gpu::Gpu,
    material::MaterialAtlas,
    mesh::{Mesh, MeshVertexArrayType},
    scene::{GpuScene, Instance},
    scene_uniform::SceneUniform,
    shader_compiler::ShaderCompiler,
};

pub struct GBuffers {
    pub g_normal: wgpu::Texture,
    pub g_diffuse: wgpu::Texture,
    pub g_specular: wgpu::Texture,
}

struct Pipelines {
    solid: wgpu::RenderPipeline,
    textured: wgpu::RenderPipeline,
    textured_normal: wgpu::RenderPipeline,
}

pub struct GeometryPass {
    g_buffers: GBuffers,
    pipelines: Pipelines,
}

impl GBuffers {
    fn new(gpu: &Gpu) -> Self {
        let viewport_size = gpu.viewport_size();

        let t_normal = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("GeometryPass::Normal"),
            size: viewport_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let t_diffuse = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("GeometryPass::Diffuse"),
            size: viewport_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let t_specular = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("GeometryPass::Specular"),
            size: viewport_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        Self {
            g_normal: t_normal,
            g_diffuse: t_diffuse,
            g_specular: t_specular,
        }
    }

    fn color_target_spec() -> &'static [Option<wgpu::ColorTargetState>] {
        &[
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
        ]
    }
}

impl Pipelines {
    pub fn new(
        gpu: &Gpu,
        shader_compiler: &mut ShaderCompiler,
        material_atlas: &MaterialAtlas,
        scene_uniform: &SceneUniform,
    ) -> Result<Self> {
        let solid_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GeometryPass::SolidPipelineLayout"),
                bind_group_layouts: &[scene_uniform.layout(), &material_atlas.layouts.phong_solid],
                push_constant_ranges: &[],
            });

        let textured_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GeometryPass::TexturedPipelineLayout"),
                bind_group_layouts: &[
                    scene_uniform.layout(),
                    &material_atlas.layouts.phong_textured,
                ],
                push_constant_ranges: &[],
            });

        let textured_normal_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("GeometryPass::TexturedNormalPipelineLayout"),
                    bind_group_layouts: &[
                        scene_uniform.layout(),
                        &material_atlas.layouts.phong_textured_normal,
                    ],
                    push_constant_ranges: &[],
                });

        let solid_shader = gpu.shader_from_module(shader_compiler.compile(
            "./shaders/deferred/geometry.wgsl",
            vec![
                ("VERTEX_PN".to_owned(), ShaderDefValue::Bool(true)),
                ("DEFERRED".to_owned(), ShaderDefValue::Bool(true)),
                (
                    "MATERIAL_PHONG_SOLID".to_owned(),
                    ShaderDefValue::Bool(true),
                ),
            ],
        )?);

        let textured_shader = gpu.shader_from_module(shader_compiler.compile(
            "./shaders/deferred/geometry.wgsl",
            vec![
                ("VERTEX_PNUV".to_owned(), ShaderDefValue::Bool(true)),
                ("DEFERRED".to_owned(), ShaderDefValue::Bool(true)),
                (
                    "MATERIAL_PHONG_TEXTURED".to_owned(),
                    ShaderDefValue::Bool(true),
                ),
            ],
        )?);

        let textured_normal_shader = gpu.shader_from_module(shader_compiler.compile(
            "./shaders/deferred/geometry.wgsl",
            vec![
                ("VERTEX_PNTBUV".to_owned(), ShaderDefValue::Bool(true)),
                ("DEFERRED".to_owned(), ShaderDefValue::Bool(true)),
                (
                    "MATERIAL_PHONG_TEXTURED".to_owned(),
                    ShaderDefValue::Bool(true),
                ),
                ("NORMAL_MAP".to_owned(), ShaderDefValue::Bool(true)),
            ],
        )?);

        let solid_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("GeometryPass::SolidPipeline"),
                layout: Some(&solid_layout),
                vertex: wgpu::VertexState {
                    module: &solid_shader,
                    entry_point: "vs_main",
                    buffers: &[
                        Mesh::pn_vertex_layout(),
                        Instance::pn_model_instance_layout(),
                    ],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &solid_shader,
                    entry_point: "fs_main",
                    targets: GBuffers::color_target_spec(),
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

        let textured_pipeline =
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("GeometryPass::TexturedPipeline"),
                    layout: Some(&textured_layout),
                    vertex: wgpu::VertexState {
                        module: &textured_shader,
                        entry_point: "vs_main",
                        buffers: &[
                            Mesh::pnuv_vertex_layout(),
                            Instance::pnuv_model_instance_layout(),
                        ],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &textured_shader,
                        entry_point: "fs_main",
                        targets: GBuffers::color_target_spec(),
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

        let textured_normal_pipeline =
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("GeometryPass::TexturedNormalPipeline"),
                    layout: Some(&textured_normal_layout),
                    vertex: wgpu::VertexState {
                        module: &textured_normal_shader,
                        entry_point: "vs_main",
                        buffers: &[
                            Mesh::pntbuv_vertex_layout(),
                            Instance::pntbuv_model_instance_layout(),
                        ],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &textured_normal_shader,
                        entry_point: "fs_main",
                        targets: GBuffers::color_target_spec(),
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
            solid: solid_pipeline,
            textured: textured_pipeline,
            textured_normal: textured_normal_pipeline,
        })
    }
}

impl GeometryPass {
    pub fn new(
        gpu: &Gpu,
        shader_compiler: &mut ShaderCompiler,
        material_atlas: &MaterialAtlas,
        scene_uniform: &SceneUniform,
    ) -> Result<Self> {
        let g_buffers = GBuffers::new(gpu);
        let pipelines = Pipelines::new(gpu, shader_compiler, material_atlas, scene_uniform)?;

        Ok(Self {
            g_buffers,
            pipelines,
        })
    }

    pub fn render(
        &self,
        gpu: &Gpu,
        atlas: &MaterialAtlas,
        scene_uniform: &SceneUniform,
        scene: &GpuScene,
    ) -> &GBuffers {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GeometryPass::CommandEncoder"),
            });

        let tv_normal = self
            .g_buffers
            .g_normal
            .create_view(&wgpu::TextureViewDescriptor::default());

        let tv_diffuse = self
            .g_buffers
            .g_diffuse
            .create_view(&wgpu::TextureViewDescriptor::default());

        let tv_specular = self
            .g_buffers
            .g_specular
            .create_view(&wgpu::TextureViewDescriptor::default());

        let tv_depth = gpu.depth_texture_view();

        {
            let mut rpass: wgpu::RenderPass<'_> =
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("GeometryPass::RenderPass"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: &tv_normal,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &tv_diffuse,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &tv_specular,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &tv_depth,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });

            for draw_call in scene.draw_calls() {
                match draw_call.vertex_array_type {
                    MeshVertexArrayType::PNUV => rpass.set_pipeline(&self.pipelines.textured),
                    MeshVertexArrayType::PNTBUV => {
                        rpass.set_pipeline(&self.pipelines.textured_normal)
                    }
                    MeshVertexArrayType::PN => rpass.set_pipeline(&self.pipelines.solid),
                };

                rpass.set_bind_group(0, scene_uniform.bind_group(), &[]);
                rpass.set_bind_group(1, atlas.bind_group(draw_call.material_id), &[]);

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
        &self.g_buffers
    }
}
