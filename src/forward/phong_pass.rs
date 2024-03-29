use std::sync::Arc;

use crate::{
    mesh::{Mesh, MeshVertexArrayType},
    render_context::RenderContext,
    scene::Instance,
};
use anyhow::Result;
use encase::{ShaderType, StorageBuffer};

pub struct PhongPass<'window> {
    render_ctx: Arc<RenderContext<'window>>,
    lights_bg: wgpu::BindGroup,
    #[allow(dead_code)]
    lights_buf: wgpu::Buffer,
    pipelines: PhongPipelines,
}

struct PhongPipelines {
    solid: wgpu::RenderPipeline,
    textured: wgpu::RenderPipeline,
    textured_normal: wgpu::RenderPipeline,
}

impl<'window> PhongPass<'window> {
    pub fn new(
        render_ctx: Arc<RenderContext<'window>>,
        shadow_bgl: &wgpu::BindGroupLayout,
    ) -> Result<Self> {
        let RenderContext {
            gpu,
            shader_compiler,
            scene_uniform,
            light_scene: lights,
            material_atlas,
            gpu_scene,
            ..
        } = render_ctx.as_ref();

        use wgpu::util::DeviceExt;

        let gpu_lights = lights.into_gpu();
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

        let module = shader_compiler
            .compilation_unit("./shaders/forward/phong.wgsl")?
            .with_def("SHADOW_MAP");

        let solid_shader =
            gpu.shader_from_module(module.compile(&["VERTEX_PN", "MATERIAL_PHONG_SOLID"])?);

        let textured_shader =
            gpu.shader_from_module(module.compile(&["VERTEX_PNUV", "MATERIAL_PHONG_TEXTURED"])?);

        let textured_normal_shader = gpu.shader_from_module(module.compile(&[
            "VERTEX_PNTBUV",
            "MATERIAL_PHONG_TEXTURED",
            "NORMAL_MAP",
        ])?);

        // Lights buffer:
        let lights_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let lights_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &lights_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_buf.as_entire_binding(),
            }],
        });

        let solid_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    scene_uniform.layout(),
                    &lights_bgl,
                    &material_atlas.layouts.phong_solid,
                    &shadow_bgl,
                ],
                push_constant_ranges: &[],
            });

        let textured_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    scene_uniform.layout(),
                    &lights_bgl,
                    &material_atlas.layouts.phong_textured,
                    &shadow_bgl,
                ],
                push_constant_ranges: &[],
            });

        let textured_normal_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[
                        scene_uniform.layout(),
                        &lights_bgl,
                        &material_atlas.layouts.phong_textured_normal,
                        &shadow_bgl,
                    ],
                    push_constant_ranges: &[],
                });

        let pipeline_solid = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
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

        let pipeline_textured =
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: None,
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

        let pipeline_textured_normal =
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: None,
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

        let pipelines = PhongPipelines {
            solid: pipeline_solid,
            textured: pipeline_textured,
            textured_normal: pipeline_textured_normal,
        };

        Ok(Self {
            render_ctx,
            lights_bg,
            lights_buf: light_buf,
            pipelines,
        })
    }

    pub fn render(&self, shadow_bg: &wgpu::BindGroup, with_prepass: bool) -> wgpu::SurfaceTexture {
        let RenderContext {
            gpu,
            scene_uniform,
            gpu_scene: scene,
            material_atlas: atlas,
            ..
        } = self.render_ctx.as_ref();

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
                        load: if with_prepass {
                            wgpu::LoadOp::Load
                        } else {
                            wgpu::LoadOp::Clear(1.0)
                        },
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_bind_group(0, scene_uniform.bind_group(), &[]);
            rpass.set_bind_group(1, &self.lights_bg, &[]);
            rpass.set_bind_group(3, shadow_bg, &[]);

            for draw_call in scene.draw_calls() {
                match draw_call.vertex_array_type {
                    MeshVertexArrayType::PNUV => rpass.set_pipeline(&self.pipelines.textured),
                    MeshVertexArrayType::PNTBUV => {
                        rpass.set_pipeline(&self.pipelines.textured_normal)
                    }
                    MeshVertexArrayType::PN => rpass.set_pipeline(&self.pipelines.solid),
                };

                rpass.set_bind_group(2, atlas.bind_group(draw_call.material_id), &[]);

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
        frame
    }
}
