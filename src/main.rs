use anyhow::{Context, Result};
use nalgebra as na;

use winit::{
    event::*, event_loop::EventLoop, keyboard::PhysicalKey, window::Window, window::WindowBuilder,
};

use camera::Camera;

use std::{borrow::Cow, time::Instant};

mod camera;
mod model;
mod world_model;

use world_model::{GpuWorldModel, WorldModel};

// OpenGL Normalized Coordinate System (right-handed):
// x: [-1.0, 1.0]
// y: [-1.0, 1.0]
// z: [-1.0, 1.0]

// WebGPU Normalized Coordinate System (left-handed):
// x: [-1.0, 1.0]
// y: [-1.0, 1.0]
// z: [0.0, 1.0]

// So we're basically scaling z by 0.5 (m33) so it becomes [-0.5, 0.5] and then translate by 0.5 (m34) so it becomes [0.0, 1.0] that WGPU expects.
#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: na::Matrix4<f32> = na::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

const MOVE_DELTA: f32 = 0.25;
const TILT_DELTA: f32 = 1.0;

// Scale matrix:
// [sx 0 0 0]
// [0 sy 0 0]
// [0 0 sz 0]
// [0 0  0 1]

// Translation matrix:
// [1 0 0 tx]
// [0 1 0 ty]
// [0 0 1 tz]
// [0 0 0 1]

use model::{Cube, GpuModel, ObjParser, Plane};

async fn run(event_loop: EventLoop<()>, window: Window) -> Result<()> {
    let mut size = window.inner_size();
    size.width = size.width.max(1);
    size.height = size.height.max(1);

    let aspect_ratio = size.width as f32 / size.height as f32;

    let instance = wgpu::Instance::default();
    let surface = unsafe { instance.create_surface(&window).unwrap() };

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            // Request an adapter which can render to our surface
            compatible_surface: Some(&surface),
        })
        .await
        .ok_or(anyhow::anyhow!("Failed to find an appropriate adapter"))?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
                limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
            },
            None,
        )
        .await
        .context("Failed to create device")?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/phong.wgsl"))),
    });

    let mut cubes = WorldModel::new(Cube::new().model());
    let mut planes = WorldModel::new(Plane::new(na::Vector3::z()).model());
    let mut teapots = WorldModel::new(ObjParser::read_model("./models/teapot.obj")?);

    planes.add(
        na::Matrix4::new_translation(&na::Vector3::new(0.0, 0.0, 0.0))
            * na::Matrix4::new_rotation(na::Vector3::x() * 270.0f32.to_radians())
            * na::Matrix4::new_scaling(100.0),
        na::Vector3::new(0.6, 0.6, 0.6),
    );

    teapots.add(
        na::Matrix4::new_translation(&na::Vector3::new(-2.0, 0.0, 0.0))
            * na::Matrix4::new_rotation(na::Vector3::y() * 33.0f32.to_radians())
            * na::Matrix4::new_scaling(2.0),
        na::Vector3::new(0.5, 0.5, 1.0),
    );

    cubes.add(
        na::Matrix4::new_translation(&na::Vector3::new(6.0, 2.001, 0.0))
            * na::Matrix4::new_rotation(na::Vector3::y() * 45.0f32.to_radians())
            * na::Matrix4::new_scaling(1.0),
        na::Vector3::new(0.8, 0.2, 0.2),
    );

    let light_idx = cubes.add(
        na::Matrix4::new_translation(&na::Vector3::new(20.0, 8.0, 0.0))
            * na::Matrix4::new_scaling(0.5),
        na::Vector3::new(1.0, 1.0, 1.0),
    );

    cubes.add(
        na::Matrix4::new_translation(&na::Vector3::new(-12.0, 2.000, 8.0))
            * na::Matrix4::new_rotation(na::Vector3::y() * 77.0f32.to_radians())
            * na::Matrix4::new_scaling(2.5),
        na::Vector3::new(0.2, 0.8, 0.4),
    );

    let mut cubes = cubes.into_gpu(&device);
    let mut planes = planes.into_gpu(&device);
    let mut teapots = teapots.into_gpu(&device);

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0];

    use wgpu::util::DeviceExt;

    // Projection matrices are flipping the Z coordinate in nalgebra, see: https://nalgebra.org/docs/user_guide/projections/
    let projection = OPENGL_TO_WGPU_MATRIX
        * na::Matrix4::new_perspective(aspect_ratio, 45.0f32.to_radians(), 0.1, 100.0);

    let proj_buf: wgpu::Buffer;
    {
        let mut buf = encase::UniformBuffer::new(vec![]);
        buf.write(&(OPENGL_TO_WGPU_MATRIX * projection))?;
        proj_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: buf.into_inner().as_slice(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
    }

    let invproj_buf: wgpu::Buffer;
    {
        let mut buf = encase::UniformBuffer::new(vec![]);
        buf.write(&(OPENGL_TO_WGPU_MATRIX * projection).try_inverse().unwrap())?;
        invproj_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: buf.into_inner().as_slice(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
    }

    let mut camera: Camera = Camera::new(
        na::Point3::new(0.0, 4.0, -40.0),
        0.0f32.to_radians(),
        90.0f32.to_radians(),
    );

    let invcamera_buf: wgpu::Buffer;
    {
        let mut buf = encase::UniformBuffer::new(vec![]);
        buf.write(&camera.look_at_matrix().try_inverse().unwrap())?;
        invcamera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: buf.into_inner().as_slice(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
    }

    let camera_buf: wgpu::Buffer;
    {
        let mut buf = encase::UniformBuffer::new(vec![]);
        buf.write(&camera.look_at_matrix())?;
        camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: buf.into_inner().as_slice(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
    }

    let scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        ],
    });

    let light_model_mat = cubes.model_mat(light_idx);
    let mut light_model_mat_contents = encase::UniformBuffer::new(vec![]);
    light_model_mat_contents.write(&light_model_mat)?;

    let light_model_mat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: light_model_mat_contents.into_inner().as_slice(),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let scene_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&scene_bgl],
        push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
            targets: &[Some(swapchain_format.into())],
        }),
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
        multiview: None,
    });

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: swapchain_format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: swapchain_capabilities.alpha_modes[0],
        view_formats: vec![],
    };

    surface.configure(&device, &config);

    let mut depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let window = &window;
    let camera = &mut camera;

    let mut delta = Instant::now();
    let delta = &mut delta;

    let cubes = &mut cubes;
    let planes = &mut planes;
    let teapots = &mut teapots;

    event_loop
        .run(move |event, target| {
            // Have the closure take ownership of the resources.
            // `event_loop.run` never returns, therefore we must do this to ensure
            // the resources are properly cleaned up.
            // let _ = (&instance, &adapter, &shader, &pipeline_layout);

            use winit::keyboard::KeyCode;
            if let Event::WindowEvent {
                window_id: _,
                event,
            } = event
            {
                match event {
                    WindowEvent::Resized(new_size) => {
                        // Reconfigure the surface with the new size
                        config.width = new_size.width.max(1);
                        config.height = new_size.height.max(1);
                        surface.configure(&device, &config);
                        depth_tex = device.create_texture(&wgpu::TextureDescriptor {
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
                        let delta_t = delta.elapsed().as_secs_f32();

                        let mut light_model_mat = cubes.model_mat(light_idx);
                        let light_rotation =
                            (((delta_t / (1.0 / 60.0)) as u64 % 360) as f32).to_radians();

                        light_model_mat.column_mut(3).x = light_rotation.cos() * 18.0;
                        light_model_mat.column_mut(3).z = light_rotation.sin() * 18.0;

                        cubes.update_world(&queue, light_idx, light_model_mat);

                        let frame = surface.get_current_texture().unwrap();
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
                            rpass.set_bind_group(0, &scene_bg, &[]);

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
