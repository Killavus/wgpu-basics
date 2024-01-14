use anyhow::Result;
use encase::{ShaderSize, UniformBuffer};
use nalgebra as na;

#[rustfmt::skip]
const OPENGL_TO_WGPU_MATRIX: na::Matrix4<f32> = na::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

pub struct GpuProjection {
    buf: wgpu::Buffer,
}

const SIZE: u64 = na::Matrix4::<f32>::SHADER_SIZE.into();

impl GpuProjection {
    pub fn new(mat: na::Matrix4<f32>, device: &wgpu::Device) -> Result<Self> {
        use wgpu::util::DeviceExt;

        let mut contents = UniformBuffer::new(Vec::with_capacity(SIZE as usize));
        contents.write(&(OPENGL_TO_WGPU_MATRIX * mat))?;

        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: contents.into_inner().as_slice(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Ok(Self { buf })
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buf
    }

    pub fn update(&self, queue: &wgpu::Queue, mat: na::Matrix4<f32>) -> Result<()> {
        let mut contents = UniformBuffer::new(Vec::with_capacity(SIZE as usize));
        contents.write(&(OPENGL_TO_WGPU_MATRIX * mat))?;

        queue.write_buffer(&self.buf, 0, contents.into_inner().as_slice());
        Ok(())
    }
}
