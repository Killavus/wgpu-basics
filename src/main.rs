use anyhow::Result;
use image::EncodableLayout;
use nalgebra as na;

use postprocess_pass::{PostprocessPass, PostprocessSettings};
use shadow_pass::DirectionalShadowPass;
use skybox_pass::SkyboxPass;
use winit::{
    dpi::PhysicalPosition, event::*, event_loop::EventLoop, keyboard::PhysicalKey, window::Window,
    window::WindowBuilder,
};

use camera::Camera;

mod camera;
mod gpu;
mod light;
mod model;
mod phong_pass;
mod postprocess_pass;
mod projection;
mod shadow_pass;
mod skybox_pass;
mod world_model;

use light::Light;
use phong_pass::{PhongPass, PhongSettings};

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
        dimension: wgpu::TextureDimension::D2,
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

    teapots.add(
        na::Matrix4::new_translation(&na::Vector3::new(-2.0, 0.0, -10.0))
            * na::Matrix4::new_rotation(na::Vector3::y() * 33.0f32.to_radians())
            * na::Matrix4::new_scaling(1.0),
        na::Vector3::new(0.5, 0.5, 1.0),
    );

    teapots.add(
        na::Matrix4::new_translation(&na::Vector3::new(-6.0, 0.0, -22.0))
            * na::Matrix4::new_rotation(na::Vector3::y() * 33.0f32.to_radians())
            * na::Matrix4::new_scaling(1.0),
        na::Vector3::new(0.5, 0.5, 1.0),
    );

    cubes.add(
        na::Matrix4::new_translation(&na::Vector3::new(4.0, 4.5, -2.0))
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

    let cubes = cubes.into_gpu(&gpu.device);
    let planes = planes.into_gpu(&gpu.device);
    let teapots = teapots.into_gpu(&gpu.device);

    let projection_mat =
        na::Matrix4::new_perspective(gpu.aspect_ratio(), 45.0f32.to_radians(), 0.1, 100.0);

    let projection: GpuProjection = GpuProjection::new(projection_mat, &gpu.device)?;

    let mut camera = GpuCamera::new(
        Camera::new(
            na::Point3::new(0.0, 18.0, 14.0),
            -45.0f32.to_radians(),
            270.0f32.to_radians(),
        ),
        &gpu.device,
    )?;

    let lights = vec![
        Light::new_directional(
            na::Vector3::new(-0.5, -0.5, -0.5).normalize(),
            na::Vector3::new(1.0, 1.0, 1.0),
        ),
        // Light::new_point(
        //     na::Vector3::new(12.0, 12.0, 2.0),
        //     na::Vector3::new(0.9, 0.43, 0.11),
        //     na::Vector3::new(1.0, 0.045, 0.0075),
        // ),
        // Light::new_spot(
        //     na::Vector3::new(0.0, 5.0, 0.0),
        //     na::Vector3::new(0.0, -1.0, 0.0),
        //     na::Vector3::new(0.0, 0.0, 1.0),
        //     45.0f32.to_radians(),
        //     na::Vector3::new(1.0, 0.045, 0.0075),
        // )
    ];

    let shadow_pass = DirectionalShadowPass::new(&gpu, vec![0.2, 0.5, 1.0])?;
    let phong_pass = PhongPass::new(
        &gpu,
        &camera,
        &projection,
        &lights,
        shadow_pass.out_bind_group_layout(),
        PhongSettings {
            ambient_strength: 0.2,
            diffuse_strength: 0.6,
            specular_strength: 0.2,
            specular_coefficient: 32.0,
        },
    )?;
    let postprocess_pass =
        PostprocessPass::new(&gpu, PostprocessSettings::new(1.0, 0.0, 1.0, 0.45))?;

    let skybox_pass = SkyboxPass::new(&gpu, &projection, &camera, skybox_tex, skybox_sampler)?;

    let window: &Window = &window;

    // let mut delta = Instant::now();
    // let delta = &mut delta;

    let cubes = &cubes;
    let planes = &planes;
    let teapots = &teapots;

    let gpu = &mut gpu;

    let mut dragging = false;
    let mut drag_origin: Option<(f64, f64)> = None;

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
                        let spass_bg = shadow_pass
                            .render(
                                gpu,
                                &lights[0],
                                &camera,
                                &projection_mat,
                                &[cubes, planes, teapots],
                            )
                            .unwrap();
                        let frame = phong_pass.render(gpu, &[cubes, planes, teapots], spass_bg);
                        let frame = skybox_pass.render(gpu, frame);
                        let frame = postprocess_pass.render(gpu, frame);

                        frame.present();
                        window.request_redraw();
                    }
                    WindowEvent::MouseInput { state, button, .. } => {
                        if state.is_pressed() {
                            if let MouseButton::Left = button {
                                window
                                    .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                                    .ok();
                                window.set_cursor_visible(false);
                                dragging = true;
                            }
                        } else {
                            window
                                .set_cursor_grab(winit::window::CursorGrabMode::None)
                                .ok();
                            window.set_cursor_visible(true);
                            dragging = false;
                            drag_origin = None;
                        }
                    }
                    WindowEvent::MouseWheel {
                        delta: MouseScrollDelta::LineDelta(_, y),
                        phase,
                        ..
                    } => {
                        if phase == TouchPhase::Moved {
                            camera.update(&gpu.queue, |c| c.forwards(y)).unwrap();
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        if dragging {
                            match drag_origin {
                                Some(origin) => {
                                    let full_size = window.inner_size();
                                    let pos = (
                                        (position.x + 1.0) / full_size.width as f64,
                                        (position.y + 1.0) / full_size.height as f64,
                                    );

                                    let delta = (pos.0 - origin.0, pos.1 - origin.1);

                                    camera
                                        .update(&gpu.queue, |c| c.tilt_horizontally(delta.0 as f32))
                                        .unwrap();
                                    camera
                                        .update(&gpu.queue, |c| c.tilt_vertically(-delta.1 as f32))
                                        .unwrap();

                                    window
                                        .set_cursor_position(PhysicalPosition::new(
                                            origin.0 * full_size.width as f64,
                                            origin.1 * full_size.height as f64,
                                        ))
                                        .ok();
                                }
                                None => {
                                    let full_size = window.inner_size();
                                    let pos = (
                                        (position.x + 1.0) / full_size.width as f64,
                                        (position.y + 1.0) / full_size.height as f64,
                                    );

                                    drag_origin = Some(pos);
                                }
                            }
                        }
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
