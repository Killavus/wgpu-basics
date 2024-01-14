use anyhow::Result;
use nalgebra as na;

use winit::{
    event::*, event_loop::EventLoop, keyboard::PhysicalKey, window::Window, window::WindowBuilder,
};

use camera::Camera;

use std::time::Instant;

mod camera;
mod gpu;
mod model;
mod projection;
mod world_model;

use world_model::{GpuWorldModel, WorldModel};

const MOVE_DELTA: f32 = 0.25;
const TILT_DELTA: f32 = 1.0;

use gpu::Gpu;
use model::{Cube, GpuModel, ObjParser, Plane};

use crate::{camera::GpuCamera, projection::GpuProjection};

async fn run(event_loop: EventLoop<()>, window: Window) -> Result<()> {
    let mut gpu = Gpu::from_window(&window).await?;

    let shader = gpu.shader_from_code(include_str!("../shaders/phong.wgsl"));
    let smap_shader = gpu.shader_from_code(include_str!("../shaders/shadowMap.wgsl"));

    let mut cubes = WorldModel::new(Cube::new().model());
    let mut planes = WorldModel::new(Plane::new().model());
    let mut teapots = WorldModel::new(ObjParser::read_model("./models/teapot.obj")?);

    planes.add(
        na::Matrix4::new_translation(&na::Vector3::new(0.0, 0.0, -2.0))
            * na::Matrix4::new_scaling(1000.0),
        na::Vector3::new(0.6, 0.6, 0.6),
    );

    teapots.add(
        na::Matrix4::new_translation(&na::Vector3::new(0.0, 0.0, -2.0))
            * na::Matrix4::new_rotation(na::Vector3::y() * 33.0f32.to_radians())
            * na::Matrix4::new_scaling(1.0),
        na::Vector3::new(0.5, 0.5, 1.0),
    );

    cubes.add(
        na::Matrix4::new_translation(&na::Vector3::new(4.0, 0.5, -2.0))
            * na::Matrix4::new_rotation(na::Vector3::y() * 45.0f32.to_radians())
            * na::Matrix4::new_scaling(1.0),
        na::Vector3::new(0.8, 0.2, 0.2),
    );

    let light_idx: usize = cubes.add(
        na::Matrix4::new_translation(&na::Vector3::new(12.0, 12.0, 0.0))
            * na::Matrix4::new_scaling(0.5),
        na::Vector3::new(1.0, 1.0, 1.0),
    );

    cubes.add(
        na::Matrix4::new_translation(&na::Vector3::new(-6.0, 0.5, -4.0)),
        na::Vector3::new(0.2, 0.8, 0.4),
    );

    let mut cubes = cubes.into_gpu(&gpu.device);
    let mut planes = planes.into_gpu(&gpu.device);
    let mut teapots = teapots.into_gpu(&gpu.device);

    let projection = GpuProjection::new(
        na::Matrix4::new_perspective(gpu.aspect_ratio(), 45.0f32.to_radians(), 0.1, 100.0),
        &gpu.device,
    )?;

    let camera = GpuCamera::new(
        Camera::new(
            na::Point3::new(0.0, 18.0, 14.0),
            -45.0f32.to_radians(),
            270.0f32.to_radians(),
        ),
        &gpu.device,
    )?;

    let mut light_pos = na::Vector3::new(12.0, 12.0, 0.0);

    let mut scene_bg: wgpu::BindGroup;
    let scene_bgl: wgpu::BindGroupLayout;
    let mut depth_tex: wgpu::Texture;
    let mut smap_tex: wgpu::Texture;
    let pipeline_layout: wgpu::PipelineLayout;
    let render_pipeline: wgpu::RenderPipeline;
    let proj_buf: wgpu::Buffer;
    let camera_buf: wgpu::Buffer;
    let light_model_mat_buf: wgpu::Buffer;
    let smap_proj_buf: wgpu::Buffer;
    let smap_cam_buf: wgpu::Buffer;
    let smap_sampler: wgpu::Sampler;
    let smap_bgl: wgpu::BindGroupLayout;
    let smap_bg: wgpu::BindGroup;
    let smap_pipeline_layout: wgpu::PipelineLayout;
    let smap_pipeline: wgpu::RenderPipeline;

    {
        use wgpu::util::DeviceExt;
        let Gpu { ref device, .. } = gpu;

        {
            let mut buf = encase::UniformBuffer::new(vec![]);

            buf.write(&na::Matrix4::look_at_rh(
                &light_pos.into(),
                &na::Point3::new(0.0, 0.0, 0.0),
                &na::Vector3::y(),
            ))?;
            smap_cam_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: buf.into_inner().as_slice(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        }

        {
            let mut buf = encase::UniformBuffer::new(vec![]);
            buf.write(&camera.look_at_matrix())?;
            camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: buf.into_inner().as_slice(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        }

        smap_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let mut buf = encase::UniformBuffer::new(vec![]);
        buf.write(
            &(OPENGL_TO_WGPU_MATRIX
                * na::Matrix4::new_perspective(1.0, 90.0f32.to_radians(), 0.1, 100.0)),
        )?;
        smap_proj_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: buf.into_inner().as_slice(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        smap_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &smap_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: smap_cam_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: smap_proj_buf.as_entire_binding(),
                },
            ],
        });

        scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
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
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Depth,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        {
            let light_model_mat = na::Matrix4::new_translation(&light_pos);
            let mut light_model_mat_contents = encase::UniformBuffer::new(vec![]);
            light_model_mat_contents.write(&light_model_mat)?;

            light_model_mat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: light_model_mat_contents.into_inner().as_slice(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        }

        smap_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 1024,
                height: 1024,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        smap_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::ClampToBorder,
            address_mode_v: wgpu::AddressMode::ClampToBorder,
            address_mode_w: wgpu::AddressMode::ClampToBorder,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: Some(wgpu::SamplerBorderColor::OpaqueWhite),
        });

        scene_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &scene_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: proj_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: invproj_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: invcamera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: light_model_mat_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(
                        &smap_tex.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&smap_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: smap_cam_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: smap_proj_buf.as_entire_binding(),
                },
            ],
        });

        pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&scene_bgl],
            push_constant_ranges: &[],
        });

        smap_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&smap_bgl],
            push_constant_ranges: &[],
        });

        render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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

        depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: gpu.viewport_size(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        smap_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&smap_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &smap_shader,
                entry_point: "vs_main",
                buffers: &[GpuModel::vertex_layout(), GpuWorldModel::instance_layout()],
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
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
    }

    let window = &window;
    let camera = &mut camera;

    let mut delta = Instant::now();
    let delta = &mut delta;

    let cubes = &mut cubes;
    let planes = &mut planes;
    let teapots = &mut teapots;

    let gpu = &mut gpu;
    let smap_tex = &mut smap_tex;
    let scene_bg = &mut scene_bg;
    let light_pos = &mut light_pos;

    event_loop
        .run(move |event, target| {
            use winit::keyboard::KeyCode;
            if let Event::WindowEvent {
                window_id: _,
                event,
            } = event
            {
                match event {
                    WindowEvent::Resized(new_size) => {
                        // Reconfigure the surface with the new size
                        gpu.on_resize((new_size.width, new_size.height));
                        depth_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
                            label: None,
                            size: wgpu::Extent3d {
                                width: new_size.width,
                                height: new_size.height,
                                depth_or_array_layers: 1,
                            },
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: wgpu::TextureFormat::Depth32Float,
                            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                            view_formats: &[],
                        });

                        // On macos the window needs to be redrawn manually after resizing
                        window.request_redraw();
                    }
                    WindowEvent::CloseRequested => {
                        target.exit();
                    }
                    WindowEvent::RedrawRequested => {
                        let Gpu {
                            ref device,
                            ref queue,
                            ..
                        } = gpu;

                        let delta_t = delta.elapsed().as_secs_f32();

                        let light_rotation =
                            (((delta_t / (1.0 / 60.0)) as u64 % 360) as f32).to_radians();

                        light_pos.x = light_rotation.cos() * 12.0;
                        light_pos.z = light_rotation.sin() * 12.0;
                        let light_model_mat = na::Matrix4::new_translation(light_pos);

                        let frame = gpu.current_texture();
                        let view = frame
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());

                        {
                            let mut buf = encase::UniformBuffer::new(vec![]);
                            buf.write(&camera.look_at_matrix()).unwrap();
                            queue.write_buffer(&camera_buf, 0, buf.into_inner().as_slice());
                        }

                        {
                            let mut buf = encase::UniformBuffer::new(vec![]);
                            let light_cam = na::Matrix4::look_at_rh(
                                &light_pos.xyz().into(),
                                &na::Point3::new(0.0, 0.0, 0.0),
                                &na::Vector3::y(),
                            );
                            buf.write(&light_cam).unwrap();
                            queue.write_buffer(&smap_cam_buf, 0, buf.into_inner().as_slice());
                        }

                        {
                            let mut buf = encase::UniformBuffer::new(vec![]);
                            buf.write(&camera.look_at_matrix().try_inverse().unwrap())
                                .unwrap();
                            queue.write_buffer(&invcamera_buf, 0, buf.into_inner().as_slice());
                        }

                        {
                            let mut light_model_mat_contents = encase::UniformBuffer::new(vec![]);
                            light_model_mat_contents.write(&light_model_mat).unwrap();
                            queue.write_buffer(
                                &light_model_mat_buf,
                                0,
                                light_model_mat_contents.into_inner().as_slice(),
                            );
                        }

                        let mut encoder = device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                        {
                            let smap_view =
                                smap_tex.create_view(&wgpu::TextureViewDescriptor::default());

                            let mut smappass =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: None,
                                    color_attachments: &[],
                                    depth_stencil_attachment: Some(
                                        wgpu::RenderPassDepthStencilAttachment {
                                            view: &smap_view,
                                            depth_ops: Some(wgpu::Operations {
                                                load: wgpu::LoadOp::Clear(1.0),
                                                store: wgpu::StoreOp::Store,
                                            }),
                                            stencil_ops: None,
                                        },
                                    ),
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                });

                            smappass.set_pipeline(&smap_pipeline);
                            smappass.set_bind_group(0, &smap_bg, &[]);

                            cubes.draw(&mut smappass);
                            planes.draw(&mut smappass);
                            teapots.draw(&mut smappass);
                        }

                        {
                            let depth_view =
                                depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

                            let mut rpass =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: None,
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: &view,
                                        resolve_target: None,
                                        ops: wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                            store: wgpu::StoreOp::Store,
                                        },
                                    })],
                                    depth_stencil_attachment: Some(
                                        wgpu::RenderPassDepthStencilAttachment {
                                            view: &depth_view,
                                            depth_ops: Some(wgpu::Operations {
                                                load: wgpu::LoadOp::Clear(1.0),
                                                store: wgpu::StoreOp::Store,
                                            }),
                                            stencil_ops: None,
                                        },
                                    ),
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                });
                            rpass.set_pipeline(&render_pipeline);
                            rpass.set_bind_group(0, scene_bg, &[]);

                            cubes.draw(&mut rpass);
                            planes.draw(&mut rpass);
                            teapots.draw(&mut rpass);
                        }

                        queue.submit(Some(encoder.finish()));
                        frame.present();
                        window.request_redraw();
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        if event.state.is_pressed() {
                            match event.physical_key {
                                PhysicalKey::Code(KeyCode::KeyA) => {
                                    camera.strafe(-MOVE_DELTA);
                                }
                                PhysicalKey::Code(KeyCode::KeyD) => {
                                    camera.strafe(MOVE_DELTA);
                                }
                                PhysicalKey::Code(KeyCode::KeyQ) => {
                                    camera.fly(MOVE_DELTA);
                                }
                                PhysicalKey::Code(KeyCode::KeyZ) => {
                                    camera.fly(-MOVE_DELTA);
                                }
                                PhysicalKey::Code(KeyCode::KeyW) => {
                                    camera.forwards(MOVE_DELTA);
                                }
                                PhysicalKey::Code(KeyCode::KeyS) => camera.forwards(-MOVE_DELTA),
                                PhysicalKey::Code(KeyCode::ArrowLeft) => {
                                    camera.tilt_horizontally(-TILT_DELTA.to_radians());
                                }
                                PhysicalKey::Code(KeyCode::ArrowRight) => {
                                    camera.tilt_horizontally(TILT_DELTA.to_radians());
                                }
                                PhysicalKey::Code(KeyCode::ArrowUp) => {
                                    camera.tilt_vertically(TILT_DELTA.to_radians());
                                }
                                PhysicalKey::Code(KeyCode::ArrowDown) => {
                                    camera.tilt_vertically(-TILT_DELTA.to_radians());
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                };
            }
        })
        .unwrap();

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new().build(&event_loop)?;

    run(event_loop, window).await?;

    Ok(())
}
