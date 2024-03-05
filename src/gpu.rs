use anyhow::Result;
use encase::{ShaderSize, UniformBuffer};
use nalgebra as na;
use std::{borrow::Cow, num::NonZeroU64, path::Path};

const MAT4_SIZE: NonZeroU64 = na::Matrix4::<f32>::SHADER_SIZE;

pub struct Gpu<'window> {
    pub instance: wgpu::Instance,
    pub surface: wgpu::Surface<'window>,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub depth_tex: wgpu::Texture,
}

use winit::window::Window;

use crate::shader_compiler::CompilationUnit;

impl<'window> Gpu<'window> {
    pub async fn from_window(window: &'window Window) -> Result<Self> {
        let instance = wgpu::Instance::default();

        let surface = instance.create_surface(window)?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or(anyhow::anyhow!("No adapter found"))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: adapter.features(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await?;

        let swapchain_capabilities = surface.get_capabilities(&adapter);
        let linear_formats = [
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Bgra8Unorm,
        ];

        let swapchain_format = linear_formats
            .into_iter()
            .find(|format| swapchain_capabilities.formats.contains(format))
            .expect("failed to find suitable surface for initialization");

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format: swapchain_format,
            width: window.inner_size().width,
            height: window.inner_size().height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: swapchain_capabilities.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: surface_config.width,
                height: surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        surface.configure(&device, &surface_config);

        Ok(Gpu {
            instance,
            surface,
            adapter,
            device,
            queue,
            surface_config,
            depth_tex,
        })
    }

    pub fn on_resize(&mut self, new_size: (u32, u32)) {
        self.surface_config.width = new_size.0;
        self.surface_config.height = new_size.1;
        self.surface.configure(&self.device, &self.surface_config);
        self.depth_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: new_size.0,
                height: new_size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
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

    pub fn shader_from_module(&self, module: wgpu::naga::Module) -> wgpu::ShaderModule {
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
            })
    }

    pub fn shader_per_vertex_type(
        &self,
        module: &CompilationUnit,
    ) -> Result<(wgpu::ShaderModule, wgpu::ShaderModule, wgpu::ShaderModule)> {
        Ok((
            self.shader_from_module(module.compile(&["VERTEX_PN"])?),
            self.shader_from_module(module.compile(&["VERTEX_PNUV"])?),
            self.shader_from_module(module.compile(&["VERTEX_PNTBUV"])?),
        ))
    }

    pub fn aspect_ratio(&self) -> f32 {
        self.surface_config.width as f32 / self.surface_config.height as f32
    }

    pub fn current_texture(&self) -> wgpu::SurfaceTexture {
        self.surface
            .get_current_texture()
            .expect("Failed to acquire next swap chain texture!")
    }

    pub fn depth_texture_view(&self) -> wgpu::TextureView {
        self.depth_tex
            .create_view(&wgpu::TextureViewDescriptor::default())
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

pub struct GpuMat4(na::Matrix4<f32>, wgpu::Buffer);

impl GpuMat4 {
    pub fn new(mat: na::Matrix4<f32>, device: &wgpu::Device) -> Result<Self> {
        use wgpu::util::DeviceExt;

        let size: u64 = MAT4_SIZE.into();
        let mut contents = UniformBuffer::new(Vec::with_capacity(size as usize));
        contents.write(&mat)?;

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: contents.into_inner().as_slice(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Ok(Self(mat, buffer))
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.1
    }

    pub fn update_with<F>(&mut self, queue: &wgpu::Queue, updater: F) -> Result<()>
    where
        F: Fn(&mut na::Matrix4<f32>),
    {
        updater(&mut self.0);

        let size: u64 = MAT4_SIZE.into();
        let mut contents = UniformBuffer::new(Vec::with_capacity(size as usize));
        contents.write(&self.0)?;

        queue.write_buffer(&self.1, 0, contents.into_inner().as_slice());
        Ok(())
    }

    pub fn update(&mut self, queue: &wgpu::Queue, mat: na::Matrix4<f32>) -> Result<()> {
        self.0 = mat;
        let size: u64 = MAT4_SIZE.into();
        let mut contents = UniformBuffer::new(Vec::with_capacity(size as usize));
        contents.write(&self.0)?;

        queue.write_buffer(&self.1, 0, contents.into_inner().as_slice());
        Ok(())
    }
}
