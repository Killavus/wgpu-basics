use egui::ComboBox;

use crate::{deferred::DeferredDebug, postprocess_pass::PostprocessSettings};

#[derive(Debug, Default, PartialEq, Eq)]
pub enum PipelineType {
    Forward,
    #[default]
    Deferred,
}

#[derive(Default)]
pub struct AppSettings {
    pub skybox_disabled: bool,
    pub depth_prepass_enabled: bool,
    postprocess: PostprocessSettings,
    pub pipeline_type: PipelineType,
    pub postprocess_disabled: bool,
    pub ssao: SsaoSettings,
    pub deferred_dbg: DeferredDebugState,
}

#[derive(Default, PartialEq, Eq)]
pub struct DeferredDebugState {
    pub enabled: bool,
    pub debug_type: DeferredDebug,
}

pub struct SsaoSettings {
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
