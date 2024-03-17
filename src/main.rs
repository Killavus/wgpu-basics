use anyhow::Result;
use egui::ComboBox;
use image::EncodableLayout;

use postprocess_pass::{PostprocessPass, PostprocessSettings};
use scene::GpuScene;
use scene_uniform::SceneUniform;
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
mod gpu_context;
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
mod ui_pass;

use forward::DepthPrepass;

const MOVE_DELTA: f32 = 1.0;
const TILT_DELTA: f32 = 1.0;

use gpu::Gpu;

use crate::phong_light::PhongLight;
use deferred::{GeometryPass, SsaoPass};

#[derive(Debug, Default, PartialEq, Eq)]
enum PipelineType {
    Forward,
    #[default]
    Deferred,
}

#[derive(Default)]
struct AppSettings {
    skybox_disabled: bool,
    depth_prepass_enabled: bool,
    deferred_enabled: bool,
    postprocess: PostprocessSettings,
    pipeline_type: PipelineType,
    postprocess_disabled: bool,
    ssao: SsaoSettings,
    deferred_dbg: DeferredDebugState,
}

#[derive(Default, PartialEq, Eq)]
enum DeferredDebug {
    #[default]
    Normals,
    Diffuse,
    Specular,
    Depth,
    AmbientOcclusion,
}

#[derive(Default)]
struct DeferredDebugState {
    enabled: bool,
    debug_type: DeferredDebug,
}

struct SsaoSettings {
    enabled: bool,
    num_samples: u32,
    radius: f32,
    blur_filter_size: u32,
    blur_iterations: u32,
}

impl Default for SsaoSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            num_samples: 64,
            radius: 0.5,
            blur_filter_size: 4,
            blur_iterations: 1,
        }
    }
}

impl AppSettings {
    pub fn render(&mut self, ctx: &egui::Context, time_delta: f32) {
        egui::Window::new("General")
            .resizable(false)
            .show(ctx, |ui| {
                ui.label("Pipeline Type");
                ComboBox::from_label("")
                    .selected_text(match self.pipeline_type {
                        PipelineType::Forward => "Forward",
                        PipelineType::Deferred => "Deferred",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.pipeline_type,
                            PipelineType::Forward,
                            "Forward",
                        );
                        ui.selectable_value(
                            &mut self.pipeline_type,
                            PipelineType::Deferred,
                            "Deferred",
                        );
                    });

                ui.checkbox(&mut self.skybox_disabled, "Disable Skybox");
                ui.checkbox(&mut self.postprocess_disabled, "Disable Postprocess");
            });

        if self.pipeline_type == PipelineType::Deferred {
            egui::Window::new("SSAO")
                .default_open(false)
                .show(ctx, |ui| {
                    ui.checkbox(&mut self.ssao.enabled, "Enable");
                    ui.label("Kernel Size");
                    ui.add(
                        egui::DragValue::new(&mut self.ssao.num_samples)
                            .speed(1)
                            .clamp_range(4..=256),
                    );
                    ui.label("Radius");
                    ui.add(
                        egui::DragValue::new(&mut self.ssao.radius)
                            .speed(0.01)
                            .clamp_range(0.0..=100.0),
                    );
                    ui.label("Blur Filter Size");
                    ui.add(
                        egui::DragValue::new(&mut self.ssao.blur_filter_size)
                            .speed(1)
                            .clamp_range(2..=128),
                    );
                    ui.label("Blur Iterations");
                    ui.add(
                        egui::DragValue::new(&mut self.ssao.blur_iterations)
                            .speed(1)
                            .clamp_range(1..=100),
                    );
                });

            egui::Window::new("Debug")
                .default_open(false)
                .show(ctx, |ui| {
                    ui.checkbox(&mut self.deferred_dbg.enabled, "Enable");
                    ui.label("Debug Type");
                    ComboBox::from_label("")
                        .selected_text(match self.deferred_dbg.debug_type {
                            DeferredDebug::Normals => "Normals",
                            DeferredDebug::Diffuse => "Diffuse",
                            DeferredDebug::Specular => "Specular",
                            DeferredDebug::Depth => "Depth",
                            DeferredDebug::AmbientOcclusion => "Ambient Occlusion",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.deferred_dbg.debug_type,
                                DeferredDebug::Normals,
                                "Normals",
                            );
                            ui.selectable_value(
                                &mut self.deferred_dbg.debug_type,
                                DeferredDebug::Diffuse,
                                "Diffuse",
                            );
                            ui.selectable_value(
                                &mut self.deferred_dbg.debug_type,
                                DeferredDebug::Specular,
                                "Specular",
                            );
                            ui.selectable_value(
                                &mut self.deferred_dbg.debug_type,
                                DeferredDebug::Depth,
                                "Depth",
                            );
                            if self.ssao.enabled {
                                ui.selectable_value(
                                    &mut self.deferred_dbg.debug_type,
                                    DeferredDebug::AmbientOcclusion,
                                    "SSAO",
                                );
                            }
                        });
                });
        }

        if self.pipeline_type == PipelineType::Forward {
            egui::Window::new("Forward")
                .default_open(false)
                .show(ctx, |ui| {
                    ui.checkbox(&mut self.depth_prepass_enabled, "Do Depth Prepass");
                });
        }

        egui::Window::new("Postprocess")
            .default_open(false)
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
    }

    pub fn postprocess_settings(&self) -> &PostprocessSettings {
        &self.postprocess
    }
}

async fn run(event_loop: EventLoop<()>, window: Window) -> Result<()> {
    let mut gpu = Gpu::from_window(&window).await?;

    let (scene, material_atlas, lights, mut camera, projection, projection_mat, _) =
        test_scenes::teapot_scene(&gpu)?;

    let mut ui_pass: UiPass = UiPass::new(&window, &gpu)?;
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

    let forward_phong_pass = forward::PhongPass::new(
        &gpu,
        &mut shader_compiler,
        &scene_uniform,
        &lights,
        &material_atlas,
        shadow_pass.out_bind_group_layout(),
    )?;

    let skybox_pass = SkyboxPass::new(
        &gpu,
        &mut shader_compiler,
        &scene_uniform,
        skybox_tex,
        skybox_sampler,
    )?;

    let geometry_pass =
        GeometryPass::new(&gpu, &mut shader_compiler, &material_atlas, &scene_uniform)?;

    let ssao_pass: SsaoPass = SsaoPass::new(&gpu, &shader_compiler, &scene_uniform)?;

    let deferred_phong_pass = deferred::PhongPass::new(
        &gpu,
        &mut shader_compiler,
        &lights,
        &scene_uniform,
        shadow_pass.out_bind_group_layout(),
    )?;

    let mut postprocess_pass = PostprocessPass::new(
        &gpu,
        &mut shader_compiler,
        &deferred_phong_pass.output_tex_view(),
        settings.postprocess_settings(),
    )?;

    let window: &Window = &window;

    let gpu = &mut gpu;

    let mut dragging = false;
    let mut drag_origin: Option<(f64, f64)> = None;

    let time = std::time::Instant::now();
    let mut last_time = time.elapsed();
    let ui = &mut ui_pass;

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

    let dbg_bgl_depth = gpu
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
                        sample_type: wgpu::TextureSampleType::Depth,
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

    let dbg_module = shader_compiler.compilation_unit("./shaders/showTexture.wgsl")?;
    let dbg_shader = gpu.shader_from_module(dbg_module.compile(&[])?);
    let dbg_depth_shader = gpu.shader_from_module(dbg_module.compile(&["DEPTH_TEXTURE"])?);

    let dbg_pipeline_l = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&dbg_bgl],
            push_constant_ranges: &[],
        });

    let dbg_pipeline_l_depth = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&dbg_bgl_depth],
            push_constant_ranges: &[],
        });

    let [dbg_pipeline, dbg_depth_pipeline] = [
        (dbg_shader, dbg_pipeline_l),
        (dbg_depth_shader, dbg_pipeline_l_depth),
    ]
    .map(|(shader, layout)| {
        gpu.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
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
            })
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

                            match settings.pipeline_type {
                                PipelineType::Deferred => {
                                    let mut frame = gpu.current_texture();

                                    let g_bufs = geometry_pass.render(
                                        gpu,
                                        &material_atlas,
                                        &scene_uniform,
                                        &gpu_scene,
                                    );

                                    let ssao_tex = ssao_pass.render(gpu, g_bufs, &scene_uniform);
                                    deferred_phong_pass.render(
                                        gpu,
                                        g_bufs,
                                        &scene_uniform,
                                        spass_bg,
                                        &ssao_tex,
                                    );

                                    if settings.deferred_dbg.enabled {
                                        let bg: wgpu::BindGroup;

                                        match settings.deferred_dbg.debug_type {
                                            DeferredDebug::Normals => {
                                                let tv = g_bufs
                                                    .g_normal
                                                    .create_view(&wgpu::TextureViewDescriptor::default());

                                                bg = gpu.device.create_bind_group(
                                                 &wgpu::BindGroupDescriptor {
                                                        label: Some("DeferredDebug::NormalsBG"),
                                                        layout: &dbg_bgl,
                                                        entries: &[
                                                            wgpu::BindGroupEntry {
                                                                binding: 0,
                                                                resource:
                                                                    wgpu::BindingResource::TextureView(&tv),
                                                            },
                                                            wgpu::BindGroupEntry {
                                                                binding: 1,
                                                                resource: wgpu::BindingResource::Sampler(
                                                                    &dbg_sampler,
                                                                ),
                                                            },
                                                        ],
                                                    },
                                                );
                                            }
                                            DeferredDebug::Diffuse => {
                                                let tv = g_bufs
                                                    .g_diffuse
                                                    .create_view(&wgpu::TextureViewDescriptor::default());

                                                bg = gpu.device.create_bind_group(
                                                    &wgpu::BindGroupDescriptor {
                                                        label: Some("DeferredDebug::DiffuseBG"),
                                                        layout: &dbg_bgl,
                                                        entries: &[
                                                            wgpu::BindGroupEntry {
                                                                binding: 0,
                                                                resource:
                                                                    wgpu::BindingResource::TextureView(&tv),
                                                            },
                                                            wgpu::BindGroupEntry {
                                                                binding: 1,
                                                                resource: wgpu::BindingResource::Sampler(
                                                                    &dbg_sampler,
                                                                ),
                                                            },
                                                        ],
                                                    },
                                                );
                                            }
                                            DeferredDebug::Specular => {
                                                let tv = g_bufs
                                                    .g_specular
                                                    .create_view(&wgpu::TextureViewDescriptor::default());

                                                bg = gpu.device.create_bind_group(
                                                    &wgpu::BindGroupDescriptor {
                                                        label: Some("DeferredDebug::SpecularBG"),
                                                        layout: &dbg_bgl,
                                                        entries: &[
                                                            wgpu::BindGroupEntry {
                                                                binding: 0,
                                                                resource:
                                                                    wgpu::BindingResource::TextureView(&tv),
                                                            },
                                                            wgpu::BindGroupEntry {
                                                                binding: 1,
                                                                resource: wgpu::BindingResource::Sampler(
                                                                    &dbg_sampler,
                                                                ),
                                                            },
                                                        ],
                                                    },
                                                );
                                            }
                                            DeferredDebug::Depth => {
                                                let tv = gpu.depth_texture_view();

                                                bg = gpu.device.create_bind_group(
                                                    &wgpu::BindGroupDescriptor {
                                                        label: Some("DeferredDebug::DepthBG"),
                                                        layout: &dbg_bgl_depth,
                                                        entries: &[
                                                            wgpu::BindGroupEntry {
                                                                binding: 0,
                                                                resource:
                                                                    wgpu::BindingResource::TextureView(&tv),
                                                            },
                                                            wgpu::BindGroupEntry {
                                                                binding: 1,
                                                                resource: wgpu::BindingResource::Sampler(
                                                                    &dbg_sampler,
                                                                ),
                                                            },
                                                        ],
                                                    },
                                                );
                                            }
                                            DeferredDebug::AmbientOcclusion => {
                                                let tv = g_bufs
                                                    .g_specular
                                                    .create_view(&wgpu::TextureViewDescriptor::default());

                                                bg = gpu.device.create_bind_group(
                                                    &wgpu::BindGroupDescriptor {
                                                        label: Some("DeferredDebug::AOBG"),
                                                        layout: &dbg_bgl,
                                                        entries: &[
                                                            wgpu::BindGroupEntry {
                                                                binding: 0,
                                                                resource:
                                                                    wgpu::BindingResource::TextureView(&tv),
                                                            },
                                                            wgpu::BindGroupEntry {
                                                                binding: 1,
                                                                resource: wgpu::BindingResource::Sampler(
                                                                    &dbg_sampler,
                                                                ),
                                                            },
                                                        ],
                                                    },
                                                );
                                            }
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

                                            if settings.deferred_dbg.debug_type == DeferredDebug::Depth {
                                                rpass.set_pipeline(&dbg_depth_pipeline);
                                            } else {
                                                rpass.set_pipeline(&dbg_pipeline);
                                            }

                                            rpass.set_bind_group(0, &bg, &[]);
                                            rpass.draw(0..4, 0..1);
                                        }

                                        gpu.queue.submit(Some(encoder.finish()));
                                    }

                                    if !settings.skybox_disabled {
                                        skybox_pass.render(gpu, &scene_uniform, deferred_phong_pass.output_tex_view(), true);
                                    }

                                    if !settings.postprocess_disabled {
                                        frame = postprocess_pass.render(
                                            gpu,
                                            settings.postprocess_settings(),
                                            frame,
                                            settings.pipeline_type == PipelineType::Deferred,
                                        );
                                    }

                                    let frame = ui.render(gpu, frame, window, ui_update);
                                    frame.present();
                                }
                                PipelineType::Forward => {
                                    if settings.depth_prepass_enabled {
                                        depth_prepass.render(gpu, &scene_uniform, &gpu_scene);
                                    }

                                    let mut frame = forward_phong_pass.render(
                                        gpu,
                                        &scene_uniform,
                                        &material_atlas,
                                        &gpu_scene,
                                        spass_bg,
                                        settings.depth_prepass_enabled,
                                    );

                                    if !settings.skybox_disabled {
                                        skybox_pass.render(gpu, &scene_uniform, frame.texture.create_view(&Default::default()), false);
                                    }

                                    if !settings.postprocess_disabled {
                                        frame = postprocess_pass.render(
                                            gpu,
                                            settings.postprocess_settings(),
                                            frame,
                                            settings.deferred_enabled,
                                        );
                                    }

                                    let frame = ui.render(gpu, frame, window, ui_update);
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
