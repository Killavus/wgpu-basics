use std::sync::Arc;

use anyhow::Result;

use postprocess_pass::PostprocessPass;
use render_context::RenderContext;
use scene::GpuScene;
use scene_uniform::SceneUniform;
use settings::AppSettings;
use shader_compiler::ShaderCompiler;
use shadow_pass::DirectionalShadowPass;
use skybox_pass::SkyboxPass;
use ui_pass::UiPass;
use winit::{
    dpi::{LogicalSize, PhysicalPosition},
    event::*,
    event_loop::EventLoop,
    keyboard::PhysicalKey,
    window::{Window, WindowBuilder},
};

mod camera;
mod compute;
mod deferred;
mod forward;
mod gpu;
mod light_scene;
mod loader;
mod material;
mod mesh;
mod postprocess_pass;
mod projection;
mod render_context;
mod scene;
mod scene_uniform;
mod settings;
mod shader_compiler;
mod shadow_pass;
mod shapes;
mod skybox_pass;
mod test_scenes;
mod ui_pass;

use forward::DepthPrepass;

const MOVE_DELTA: f32 = 1.0;
const TILT_DELTA: f32 = 1.0;

use gpu::Gpu;

use crate::{light_scene::Light, settings::PipelineType};
use deferred::{GeometryPass, SsaoPass};

async fn run(event_loop: EventLoop<()>, window: Window) -> Result<()> {
    let mut gpu = Gpu::from_window(&window).await?;

    let (scene, material_atlas, lights, mut camera, projection, projection_mat, _) =
        test_scenes::teapot_scene(&gpu)?;
    let gpu_scene = GpuScene::new(&gpu, scene)?;
    let scene_uniform = SceneUniform::new(&gpu, &camera, &projection);

    let render_ctx = Arc::new(RenderContext::new(
        &window,
        gpu,
        ShaderCompiler::new("./shaders")?,
        scene_uniform,
        gpu_scene,
        material_atlas,
        lights,
    ));

    let mut ui_pass: UiPass = UiPass::new(render_ctx.clone())?;
    let mut settings: AppSettings = AppSettings::default();

    let skybox_texture = test_scenes::load_skybox(&render_ctx.gpu)?;

    let shadow_pass =
        DirectionalShadowPass::new(render_ctx.clone(), [0.2, 0.5, 1.0], &projection_mat)?;
    let depth_prepass = DepthPrepass::new(render_ctx.clone())?;

    let forward_phong_pass =
        forward::PhongPass::new(render_ctx.clone(), shadow_pass.out_bind_group_layout())?;

    let skybox_pass = SkyboxPass::new(render_ctx.clone(), skybox_texture)?;

    let geometry_pass = GeometryPass::new(render_ctx.clone())?;

    let deferred_debug_pass = deferred::DebugPass::new(render_ctx.clone())?;

    let ssao_pass: SsaoPass = SsaoPass::new(render_ctx.clone())?;

    let deferred_phong_pass =
        deferred::PhongPass::new(render_ctx.clone(), shadow_pass.out_bind_group_layout())?;

    let postprocess_pass = PostprocessPass::new(
        render_ctx.clone(),
        &deferred_phong_pass.output_tex_view(),
        settings.postprocess_settings(),
    )?;

    let window: &Window = &window;

    let mut dragging = false;
    let mut drag_origin: Option<(f64, f64)> = None;

    let time = std::time::Instant::now();
    let mut last_time = time.elapsed();
    let ui = &mut ui_pass;

    let render_ctx = render_ctx.clone();
    event_loop
        .run(move |event, target| {
            use winit::keyboard::KeyCode;
            let gpu = &render_ctx.gpu;
            let lights = &render_ctx.light_scene;

            if let Event::WindowEvent {
                window_id: _,
                event,
            } = event
            {
                if !ui.handle_input(window, &event) {
                    match event {
                        WindowEvent::Resized(new_size) => {
                            // Reconfigure the surface with the new size
                            // gpu.on_resize((new_size.width, new_size.height));
                            // postprocess_pass.on_resize(gpu, (new_size.width, new_size.height));
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

                            let spass_bg = shadow_pass
                                .render(
                                    lights
                                        .directional
                                        .first()
                                        .unwrap_or(&Light::new_directional(
                                            na::Vector3::zeros(),
                                            na::Vector3::zeros(),
                                            na::Vector3::zeros(),
                                            na::Vector3::zeros(),
                                        )),
                                    &camera,
                                    &projection_mat,
                                )
                                .unwrap();

                            match settings.pipeline_type {
                                PipelineType::Deferred => {
                                    let mut frame = gpu.current_texture();

                                    let g_bufs = geometry_pass.render();

                                    let ssao_tex = ssao_pass.render(g_bufs);

                                    deferred_phong_pass.render(g_bufs, spass_bg, &ssao_tex);

                                    if settings.deferred_dbg.enabled {
                                        deferred_debug_pass.render(
                                            g_bufs,
                                            &frame,
                                            &ssao_tex,
                                            &settings.deferred_dbg.debug_type,
                                        )
                                    } else {
                                        if !settings.skybox_disabled {
                                            skybox_pass.render(
                                                deferred_phong_pass.output_tex_view(),
                                                true,
                                            );
                                        }

                                        if !settings.postprocess_disabled {
                                            frame = postprocess_pass.render(
                                                settings.postprocess_settings(),
                                                frame,
                                                settings.pipeline_type == PipelineType::Deferred,
                                            );
                                        }
                                    }

                                    let frame = ui.render(frame, ui_update);
                                    frame.present();
                                }
                                PipelineType::Forward => {
                                    if settings.depth_prepass_enabled {
                                        depth_prepass.render();
                                    }

                                    let mut frame = forward_phong_pass
                                        .render(spass_bg, settings.depth_prepass_enabled);

                                    if !settings.skybox_disabled {
                                        skybox_pass.render(
                                            frame.texture.create_view(&Default::default()),
                                            false,
                                        );
                                    }

                                    if !settings.postprocess_disabled {
                                        frame = postprocess_pass.render(
                                            settings.postprocess_settings(),
                                            frame,
                                            settings.pipeline_type == PipelineType::Deferred,
                                        );
                                    }

                                    let frame = ui.render(frame, ui_update);
                                    frame.present();
                                }
                            }

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
