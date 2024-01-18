use anyhow::Result;
use image::EncodableLayout;
use nalgebra as na;

use winit::{
    event::*, event_loop::EventLoop, keyboard::PhysicalKey, window::Window, window::WindowBuilder,
};

use camera::Camera;

use std::time::Instant;

mod camera;
mod gpu;
mod model;
mod phong_pass;
mod projection;
mod world_model;

use phong_pass::{Light, PhongPass};

use world_model::WorldModel;

const MOVE_DELTA: f32 = 0.25;
const TILT_DELTA: f32 = 1.0;

use gpu::Gpu;
use model::{Cube, ObjParser, Plane};

use crate::{camera::GpuCamera, projection::GpuProjection};

async fn run(event_loop: EventLoop<()>, window: Window) -> Result<()> {
    let mut gpu = Gpu::from_window(&window).await?;

    let mut cubes = WorldModel::new(Cube::new().model());
    let mut planes = WorldModel::new(Plane::new().model());
    let mut teapots = WorldModel::new(ObjParser::read_model("./models/teapot.obj")?);

    // [+X, -X, +Y, -Y, +Z, -Z]
    let (sky_width, sky_height, sky_data) = [
        image::open("./textures/skybox/posx.jpg")?,
        image::open("./textures/skybox/negx.jpg")?,
        image::open("./textures/skybox/posy.jpg")?,
        image::open("./textures/skybox/negy.jpg")?,
        image::open("./textures/skybox/posz.jpg")?,
        image::open("./textures/skybox/negz.jpg")?,
    ]
    .into_iter()
    .fold((0, 0, vec![]), |mut acc, img| {
        acc.0 = img.width();
        acc.1 = img.height();
        acc.2.extend_from_slice(img.to_rgba8().as_bytes());

        acc
    });

    let skybox_size = wgpu::Extent3d {
        width: sky_width,
        height: sky_height,
        depth_or_array_layers: 6,
    };

    let skybox_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: skybox_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    gpu.queue.write_texture(
        skybox_tex.as_image_copy(),
        &sky_data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * sky_width),
            rows_per_image: Some(sky_height),
        },
        skybox_size,
    );

    let skybox_tex_view: wgpu::TextureView = skybox_tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });

    let skybox_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
        label: None,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

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

    cubes.add(
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

    let mut camera = GpuCamera::new(
        Camera::new(
            na::Point3::new(0.0, 18.0, 14.0),
            -45.0f32.to_radians(),
            270.0f32.to_radians(),
        ),
        &gpu.device,
    )?;

    let mut lights = Vec::new();
    lights.push(Light::new_point(
        na::Vector3::new(12.0, 12.0, 2.0),
        na::Vector3::new(1.0, 1.0, 1.0),
    ));

    let phong_pass = PhongPass::new(&gpu, &camera, &projection, &lights)?;
    let window = &window;

    let mut delta = Instant::now();
    let delta = &mut delta;

    let cubes = &cubes;
    let planes = &planes;
    let teapots = &teapots;

    let gpu = &mut gpu;

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
                        window.request_redraw();
                    }
                    WindowEvent::CloseRequested => {
                        target.exit();
                    }
                    WindowEvent::RedrawRequested => {
                        phong_pass.render(gpu, &[cubes, planes, teapots]);
                        window.request_redraw();
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        if event.state.is_pressed() {
                            match event.physical_key {
                                PhysicalKey::Code(KeyCode::KeyA) => {
                                    camera
                                        .update(&gpu.queue, |c| c.strafe(-MOVE_DELTA))
                                        .unwrap();
                                }
                                PhysicalKey::Code(KeyCode::KeyD) => {
                                    camera.update(&gpu.queue, |c| c.strafe(MOVE_DELTA)).unwrap();
                                }
                                PhysicalKey::Code(KeyCode::KeyQ) => {
                                    camera.update(&gpu.queue, |c| c.fly(MOVE_DELTA)).unwrap();
                                }
                                PhysicalKey::Code(KeyCode::KeyZ) => {
                                    camera.update(&gpu.queue, |c| c.fly(-MOVE_DELTA)).unwrap();
                                }
                                PhysicalKey::Code(KeyCode::KeyW) => {
                                    camera
                                        .update(&gpu.queue, |c| c.forwards(MOVE_DELTA))
                                        .unwrap();
                                }
                                PhysicalKey::Code(KeyCode::KeyS) => {
                                    camera
                                        .update(&gpu.queue, |c| c.forwards(-MOVE_DELTA))
                                        .unwrap();
                                }
                                PhysicalKey::Code(KeyCode::ArrowLeft) => {
                                    camera
                                        .update(&gpu.queue, |c| {
                                            c.tilt_horizontally(-TILT_DELTA.to_radians())
                                        })
                                        .unwrap();
                                }
                                PhysicalKey::Code(KeyCode::ArrowRight) => {
                                    camera
                                        .update(&gpu.queue, |c| {
                                            c.tilt_horizontally(TILT_DELTA.to_radians())
                                        })
                                        .unwrap();
                                }
                                PhysicalKey::Code(KeyCode::ArrowUp) => {
                                    camera
                                        .update(&gpu.queue, |c| {
                                            c.tilt_vertically(TILT_DELTA.to_radians())
                                        })
                                        .unwrap();
                                }
                                PhysicalKey::Code(KeyCode::ArrowDown) => {
                                    camera
                                        .update(&gpu.queue, |c| {
                                            c.tilt_vertically(-TILT_DELTA.to_radians())
                                        })
                                        .unwrap();
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
