use std::{borrow::Cow, path::Path};
pub struct Gpu {
    pub instance: wgpu::Instance,
    pub surface: wgpu::Surface,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface_config: wgpu::SurfaceConfiguration,
}

use anyhow::Result;
use winit::window::Window;

impl Gpu {
    pub async fn from_window(window: &Window) -> Result<Self> {
        get_gpu(window).await
    }

    pub fn on_resize(&mut self, new_size: (u32, u32)) {
        self.surface_config.width = new_size.0;
        self.surface_config.height = new_size.1;
        self.surface.configure(&self.device, &self.surface_config);
    }

    pub fn viewport_size(&self) -> wgpu::Extent3d {
        wgpu::Extent3d {
            width: self.surface_config.width,
            height: self.surface_config.height,
            depth_or_array_layers: 1,
        }
    }

    pub fn shader_from_code(&self, code: &str) -> wgpu::ShaderModule {
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(code)),
            })
    }

    pub fn aspect_ratio(&self) -> f32 {
        self.surface_config.width as f32 / self.surface_config.height as f32
    }

    pub fn current_texture(&self) -> wgpu::SurfaceTexture {
        self.surface
            .get_current_texture()
            .expect("Failed to acquire next swap chain texture!")
    }

    pub fn shader_from_file(&self, path: impl AsRef<Path>) -> Result<wgpu::ShaderModule> {
        let path = path.as_ref();
        let code = std::fs::read_to_string(path)?;
        Ok(self.shader_from_code(&code))
    }

    pub fn swapchain_format(&self) -> wgpu::TextureFormat {
        self.surface_config.format
    }
}

async fn get_gpu(window: &Window) -> Result<Gpu> {
    let instance = wgpu::Instance::default();

    let surface = unsafe { instance.create_surface(&window)? };
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .map_or(Err(anyhow::anyhow!("No adapter found")), Ok)?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: adapter.features(),
                limits: wgpu::Limits::default(),
            },
            None,
        )
        .await?;

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = wgpu::TextureFormat::Rgba8UnormSrgb;

    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: swapchain_format,
        width: window.inner_size().width,
        height: window.inner_size().height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: swapchain_capabilities.alpha_modes[0],
        view_formats: vec![],
    };

    surface.configure(&device, &surface_config);

    Ok(Gpu {
        instance,
        surface,
        adapter,
        device,
        queue,
        surface_config,
    })
}
