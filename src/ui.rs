use anyhow::Result;

use crate::gpu::Gpu;

pub struct Ui {
    ctx: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
}

impl Ui {
    pub fn new(window: &winit::window::Window, gpu: &Gpu) -> Result<Self> {
        let ctx = egui::Context::default();
        let viewport_id = ctx.viewport_id();

        let mut visuals = egui::Visuals::dark();
        visuals.window_shadow.extrusion = 0.0;

        ctx.set_visuals(visuals);

        let state = egui_winit::State::new(ctx.clone(), viewport_id, window, None, None);
        let renderer = egui_wgpu::Renderer::new(&gpu.device, gpu.swapchain_format(), None, 1);

        Ok(Self {
            ctx,
            state,
            renderer,
        })
    }

    pub fn handle_input(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::WindowEvent,
    ) -> bool {
        let response = self.state.on_window_event(window, event);

        response.consumed
    }

    pub fn update<F>(&mut self, window: &winit::window::Window, ui: F) -> egui::FullOutput
    where
        F: FnOnce(&egui::Context),
    {
        let input = self.state.take_egui_input(window);

        self.ctx.run(input, ui)
    }

    pub fn render(
        &mut self,
        gpu: &Gpu,
        frame: wgpu::SurfaceTexture,
        window: &winit::window::Window,
        output: egui::FullOutput,
    ) -> wgpu::SurfaceTexture {
        self.state
            .handle_platform_output(window, output.platform_output);

        let paint_jobs = self.ctx.tessellate(output.shapes, output.pixels_per_point);
        for (tid, delta) in output.textures_delta.set {
            self.renderer
                .update_texture(&gpu.device, &gpu.queue, tid, &delta);
        }

        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [window.inner_size().width, window.inner_size().height],
            pixels_per_point: window.scale_factor() as f32,
        };

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        self.renderer
            .update_buffers(&gpu.device, &gpu.queue, &mut encoder, &paint_jobs, &screen);

        let frame_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            self.renderer.render(&mut rpass, &paint_jobs, &screen);
        }

        gpu.queue.submit(Some(encoder.finish()));
        for tid in output.textures_delta.free {
            self.renderer.free_texture(&tid);
        }

        frame
    }
}
