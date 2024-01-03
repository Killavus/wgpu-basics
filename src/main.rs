use anyhow::{Context, Result};

use na::geometry as na_g;
use nalgebra as na;

use winit::{event::*, event_loop::EventLoop, window::Window, window::WindowBuilder};

use std::{borrow::Cow, path::Path};

// OpenGL Normalized Coordinate System (right-handed):
// x: [-1.0, 1.0]
// y: [-1.0, 1.0]
// z: [-1.0, 1.0]

// WebGPU Normalized Coordinate System (left-handed):
// x: [-1.0, 1.0]
// y: [-1.0, 1.0]
// z: [0.0, 1.0]

// So we're basically scaling z by 0.5 (m33) so it becomes [-0.5, 0.5] and then translate by 0.5 (m34)
#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: na::Matrix4<f32> = na::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

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

#[derive(Debug)]
struct Model {
    vertices: Vec<na::Vector3<f32>>,
    indices: Vec<u32>,
}

fn read_obj(path: impl AsRef<Path>) -> Result<Model> {
    use std::fs::File;
    use std::io::{prelude::*, BufReader};

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let reader = BufReader::new(File::open(path)?);

    for line in reader.lines() {
        let line = line?;

        if line.is_empty() {
            continue;
        }

        match &line[0..1] {
            "v" => {
                let mut iter = line.split_whitespace();
                iter.next();
                let x = iter.next().unwrap().parse::<f32>().unwrap();
                let y = iter.next().unwrap().parse::<f32>().unwrap();
                let z = iter.next().unwrap().parse::<f32>().unwrap();
                vertices.push(na::Vector3::new(x, y, z));
            }
            "f" => {
                let mut iter = line.split_whitespace();
                iter.next();
                let x = iter.next().unwrap().parse::<u32>().unwrap();
                let y = iter.next().unwrap().parse::<u32>().unwrap();
                let z = iter.next().unwrap().parse::<u32>().unwrap();
                indices.push(x - 1);
                indices.push(y - 1);
                indices.push(z - 1);
            }
            _ => {}
        }
    }

    Ok(Model { vertices, indices })
}

fn calculate_normals(model: &Model) -> Vec<na::Vector3<f32>> {
    let mut normals = vec![na::Vector3::new(0.0, 0.0, 0.0); model.vertices.len()];

    for i in (0..model.indices.len()).step_by(3) {
        let i0 = model.indices[i] as usize;
        let i1 = model.indices[i + 1] as usize;
        let i2 = model.indices[i + 2] as usize;

        let v0 = model.vertices[i0];
        let v1 = model.vertices[i1];
        let v2 = model.vertices[i2];

        let e1 = v1 - v0;
        let e2 = v2 - v0;

        let normal = e1.cross(&e2).normalize();

        normals[i0] += normal;
        normals[i1] += normal;
        normals[i2] += normal;
    }

    normals
}

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

    let model = read_obj("./models/teapot.obj")?;

    let normals = calculate_normals(&model);
    let vertex_buf = model
        .vertices
        .iter()
        .copied()
        .zip(normals.iter().copied())
        .flat_map(|(v, n)| [v, n])
        .collect::<Vec<_>>();

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0];

    use wgpu::util::DeviceExt;

    let vertices_buf: wgpu::Buffer;
    {
        vertices_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&vertex_buf),
            usage: wgpu::BufferUsages::VERTEX,
        });
    }

    let indices_buf: wgpu::Buffer;
    {
        indices_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&model.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
    }

    // Projection matrices are flipping the Z coordinate in nalgebra, see: https://nalgebra.org/docs/user_guide/projections/
    let projection = na::Matrix4::new_perspective(aspect_ratio, 45.0f32.to_radians(), 0.1, 100.0);

    let camera = na::Matrix4::look_at_lh(
        &na::Point3::new(0.0, 0.0, 0.0),
        &na::Point3::new(0.0, 0.0, 1.0),
        &na::Vector3::new(0.0, 1.0, 0.0),
    );

    let proj_cam = OPENGL_TO_WGPU_MATRIX * projection * camera;

    let camera_buf: wgpu::Buffer;
    {
        let mut buf = encase::UniformBuffer::new(vec![]);
        buf.write(&proj_cam)?;
        camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: buf.into_inner().as_slice(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
    }

    let model_pos_buf: wgpu::Buffer;
    {
        let mut model_mat = na::Matrix4::<f32>::identity();
        model_mat *= na::Matrix4::new_translation(&na::Vector3::new(0.0, 0.0, -5.0));
        model_mat *= na::Matrix4::new_scaling(0.33);

        let mut buf = encase::UniformBuffer::new(vec![]);
        buf.write(&model_mat)?;
        model_pos_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: buf.into_inner().as_slice(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
    }

    let cam_pos_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        ],
    });

    let cam_pos_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &cam_pos_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: model_pos_buf.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&cam_pos_bgl],
        push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: (std::mem::size_of::<na::Vector3<f32>>() * 2) as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        format: wgpu::VertexFormat::Float32x3,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        offset: std::mem::size_of::<na::Vector3<f32>>() as wgpu::BufferAddress,
                        format: wgpu::VertexFormat::Float32x3,
                        shader_location: 1,
                    },
                ],
            }],
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
    event_loop
        .run(move |event, target| {
            // Have the closure take ownership of the resources.
            // `event_loop.run` never returns, therefore we must do this to ensure
            // the resources are properly cleaned up.
            // let _ = (&instance, &adapter, &shader, &pipeline_layout);

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
                    WindowEvent::RedrawRequested => {
                        let frame = surface.get_current_texture().unwrap();
                        let view = frame
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());
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
                            rpass.set_vertex_buffer(0, vertices_buf.slice(..));
                            rpass
                                .set_index_buffer(indices_buf.slice(..), wgpu::IndexFormat::Uint32);
                            rpass.set_bind_group(0, &cam_pos_bg, &[]);
                            rpass.draw_indexed(0..model.indices.len() as u32, 0, 0..1);
                        }
                        queue.submit(Some(encoder.finish()));
                        frame.present()
                    }
                    WindowEvent::CloseRequested => target.exit(),
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
