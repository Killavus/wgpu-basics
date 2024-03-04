use anyhow::Result;
use image::EncodableLayout;

use postprocess_pass::{PostprocessPass, PostprocessSettings};
use scene::GpuScene;
use scene_uniform::SceneUniform;
use shader_compiler::ShaderCompiler;
use shadow_pass::DirectionalShadowPass;
use skybox_pass::SkyboxPass;
use winit::{
    dpi::{LogicalSize, PhysicalPosition},
    event::*,
    event_loop::EventLoop,
    keyboard::PhysicalKey,
    window::{Window, WindowBuilder},
};

mod camera;
mod deferred;
mod forward;
mod gpu;
mod loader;
mod material;
mod mesh;
mod phong_light;
mod postprocess_pass;
mod projection;
mod scene;
mod scene_uniform;
mod shader_compiler;
mod shadow_pass;
mod shapes;
mod skybox_pass;
mod test_scenes;
mod ui;

use forward::{DepthPrepass, PhongPass};

const MOVE_DELTA: f32 = 1.0;
const TILT_DELTA: f32 = 1.0;

use gpu::Gpu;

use crate::phong_light::PhongLight;
use deferred::GeometryPass;

#[derive(Default)]
struct AppSettings {
    skybox_disabled: bool,
    depth_prepass_enabled: bool,
    postprocess: PostprocessSettings,
    debug_normals: bool,
    debug_diffuse: bool,
    debug_specular: bool,
}

impl AppSettings {
    pub fn render(&mut self, ctx: &egui::Context, time_delta: f32) {
        egui::Window::new("Postprocess")
            .resizable(true)
            .show(ctx, |ui| {
                ui.label("Saturation");
                ui.add(egui::DragValue::new(self.postprocess.saturation_mut()).speed(0.01));
                ui.label("Brightness");
                ui.add(egui::DragValue::new(self.postprocess.brightness_mut()).speed(0.01));
                ui.label("Contrast");
                ui.add(egui::DragValue::new(self.postprocess.contrast_mut()).speed(0.01));
                ui.label("Gamma");
                ui.add(egui::DragValue::new(self.postprocess.gamma_mut()).speed(0.01));
            });

        egui::Window::new("Info").show(ctx, |ui| {
            ui.label(format!("FPS: {:.2}", 1.0 / time_delta));
        });

        egui::Window::new("Optional passes").show(ctx, |ui| {
            ui.checkbox(&mut self.skybox_disabled, "Disable Skybox");
            ui.checkbox(&mut self.depth_prepass_enabled, "Enable Depth Prepass");
        });

        egui::Window::new("Debug").show(ctx, |ui| {
            ui.checkbox(&mut self.debug_normals, "Debug Normals");
            ui.checkbox(&mut self.debug_diffuse, "Debug Diffuse");
            ui.checkbox(&mut self.debug_specular, "Debug Specular");
        });
    }

    pub fn postprocess_settings(&self) -> &PostprocessSettings {
        &self.postprocess
    }
}

async fn run(event_loop: EventLoop<()>, window: Window) -> Result<()> {
    let mut gpu = Gpu::from_window(&window).await?;

    let (scene, material_atlas, lights, mut camera, projection, projection_mat, _) =
        test_scenes::teapot_scene(&gpu)?;

    let mut ui = ui::Ui::new(&window, &gpu)?;
    let mut settings: AppSettings = AppSettings::default();

    let gpu_scene = GpuScene::new(&gpu, scene)?;

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

    let scene_uniform = SceneUniform::new(&gpu, &camera, &projection);
    let mut shader_compiler = ShaderCompiler::new("./shaders")?;

    let shadow_pass =
        DirectionalShadowPass::new(&gpu, &mut shader_compiler, [0.2, 0.5, 1.0], &projection_mat)?;

    let depth_prepass = DepthPrepass::new(&gpu, &mut shader_compiler, &scene_uniform)?;

    let phong_pass = PhongPass::new(
        &gpu,
        &mut shader_compiler,
        &scene_uniform,
        &lights,
        &material_atlas,
        shadow_pass.out_bind_group_layout(),
    )?;

    let mut postprocess_pass =
        PostprocessPass::new(&gpu, &mut shader_compiler, settings.postprocess_settings())?;

    let skybox_pass = SkyboxPass::new(
        &gpu,
        &mut shader_compiler,
        &scene_uniform,
        skybox_tex,
        skybox_sampler,
    )?;

    let geometry_pass =
        GeometryPass::new(&gpu, &mut shader_compiler, &material_atlas, &scene_uniform)?;

    let fill_pass = deferred::FillPass::new(
        &gpu,
        &mut shader_compiler,
        &lights,
        &scene_uniform,
        geometry_pass.g_buffers(),
    )?;

    let window: &Window = &window;

    let gpu = &mut gpu;

    let mut dragging = false;
    let mut drag_origin: Option<(f64, f64)> = None;

    let time = std::time::Instant::now();
    let mut last_time = time.elapsed();
    let ui = &mut ui;

    let dbg_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
        label: None,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let dbg_bgl = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

    let dbg_shader = gpu.shader_from_file("./shaders/showTexture.wgsl")?;
    let dbg_pipeline_l = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&dbg_bgl],
            push_constant_ranges: &[],
        });
    let dbg_pipeline = gpu
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&dbg_pipeline_l),
            vertex: wgpu::VertexState {
                module: &dbg_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &dbg_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.swapchain_format(),
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

    event_loop
        .run(move |event, target| {
            use winit::keyboard::KeyCode;

            if let Event::WindowEvent {
                window_id: _,
                event,
            } = event
            {
                if !ui.handle_input(window, &event) {
                    match event {
                        WindowEvent::Resized(new_size) => {
                            // Reconfigure the surface with the new size
                            gpu.on_resize((new_size.width, new_size.height));
                            postprocess_pass.on_resize(gpu, (new_size.width, new_size.height));
                            window.request_redraw();
                        }
                        WindowEvent::CloseRequested => {
                            target.exit();
                        }
                        WindowEvent::RedrawRequested => {
                            use nalgebra as na;
                            let time = time.elapsed();

                            let time_ms = (time - last_time).as_secs_f32();
                            let ui_update = ui.update(window, |ctx| settings.render(ctx, time_ms));

                            let g_bufs = geometry_pass.render(
                                gpu,
                                &material_atlas,
                                &scene_uniform,
                                &gpu_scene,
                            );

                            fill_pass.render(gpu, &scene_uniform);

                            if settings.depth_prepass_enabled {
                                depth_prepass.render(gpu, &scene_uniform, &gpu_scene);
                            }

                            let spass_bg = shadow_pass
                                .render(
                                    gpu,
                                    lights.directional.first().unwrap_or(
                                        &PhongLight::new_directional(
                                            na::Vector3::zeros(),
                                            na::Vector3::zeros(),
                                            na::Vector3::zeros(),
                                            na::Vector3::zeros(),
                                        ),
                                    ),
                                    &camera,
                                    &projection_mat,
                                    &gpu_scene,
                                )
                                .unwrap();

                            let mut frame = phong_pass.render(
                                gpu,
                                &scene_uniform,
                                &material_atlas,
                                &gpu_scene,
                                spass_bg,
                                settings.depth_prepass_enabled,
                            );

                            if !settings.skybox_disabled {
                                frame = skybox_pass.render(gpu, &scene_uniform, frame);
                            }

                            let frame = postprocess_pass.render(
                                gpu,
                                settings.postprocess_settings(),
                                frame,
                            );

                            if settings.debug_diffuse
                                || settings.debug_normals
                                || settings.debug_specular
                            {
                                let bg: wgpu::BindGroup;
                                if settings.debug_diffuse {
                                    let tv = g_bufs
                                        .g_diffuse
                                        .create_view(&wgpu::TextureViewDescriptor::default());

                                    bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                        label: None,
                                        layout: &dbg_bgl,
                                        entries: &[
                                            wgpu::BindGroupEntry {
                                                binding: 0,
                                                resource: wgpu::BindingResource::TextureView(&tv),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 1,
                                                resource: wgpu::BindingResource::Sampler(
                                                    &dbg_sampler,
                                                ),
                                            },
                                        ],
                                    });
                                } else if settings.debug_normals {
                                    let tv = g_bufs
                                        .g_normal
                                        .create_view(&wgpu::TextureViewDescriptor::default());

                                    bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                        label: None,
                                        layout: &dbg_bgl,
                                        entries: &[
                                            wgpu::BindGroupEntry {
                                                binding: 0,
                                                resource: wgpu::BindingResource::TextureView(&tv),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 1,
                                                resource: wgpu::BindingResource::Sampler(
                                                    &dbg_sampler,
                                                ),
                                            },
                                        ],
                                    });
                                } else {
                                    let tv = g_bufs
                                        .g_specular
                                        .create_view(&wgpu::TextureViewDescriptor::default());

                                    bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                        label: None,
                                        layout: &dbg_bgl,
                                        entries: &[
                                            wgpu::BindGroupEntry {
                                                binding: 0,
                                                resource: wgpu::BindingResource::TextureView(&tv),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 1,
                                                resource: wgpu::BindingResource::Sampler(
                                                    &dbg_sampler,
                                                ),
                                            },
                                        ],
                                    });
                                }

                                let mut encoder = gpu.device.create_command_encoder(
                                    &wgpu::CommandEncoderDescriptor::default(),
                                );
                                let frame_view = frame
                                    .texture
                                    .create_view(&wgpu::TextureViewDescriptor::default());

                                {
                                    let mut rpass =
                                        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                            label: None,
                                            color_attachments: &[Some(
                                                wgpu::RenderPassColorAttachment {
                                                    view: &frame_view,
                                                    resolve_target: None,
                                                    ops: wgpu::Operations {
                                                        load: wgpu::LoadOp::Clear(
                                                            wgpu::Color::BLACK,
                                                        ),
                                                        store: wgpu::StoreOp::Store,
                                                    },
                                                },
                                            )],
                                            depth_stencil_attachment: None,
                                            timestamp_writes: None,
                                            occlusion_query_set: None,
                                        });

                                    rpass.set_pipeline(&dbg_pipeline);
                                    rpass.set_bind_group(0, &bg, &[]);
                                    rpass.draw(0..4, 0..1);
                                }

                                gpu.queue.submit(Some(encoder.finish()));
                            }

                            let frame = ui.render(gpu, frame, window, ui_update);

                            frame.present();
                            last_time = time;
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
                                            .update(&gpu.queue, |c| {
                                                c.tilt_horizontally(delta.0 as f32)
                                            })
                                            .unwrap();
                                        camera
                                            .update(&gpu.queue, |c| {
                                                c.tilt_vertically(-delta.1 as f32)
                                            })
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
                                        camera
                                            .update(&gpu.queue, |c| c.strafe(MOVE_DELTA))
                                            .unwrap();
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
            }
        })
        .unwrap();

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_inner_size(LogicalSize::new(1366, 768))
        .build(&event_loop)?;

    run(event_loop, window).await?;

    Ok(())
}
