use crate::gpu::GpuMat4;
use anyhow::Result;
use nalgebra as na;

#[rustfmt::skip]
const OPENGL_TO_WGPU_MATRIX: na::Matrix4<f32> = na::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

pub struct GpuProjection(GpuMat4);

impl GpuProjection {
    pub fn new(mat: na::Matrix4<f32>, device: &wgpu::Device) -> Result<Self> {
        Ok(Self(GpuMat4::new(OPENGL_TO_WGPU_MATRIX * mat, device)?))
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        self.0.buffer()
    }

    pub fn update(&mut self, queue: &wgpu::Queue, mat: na::Matrix4<f32>) -> Result<()> {
        self.0.update(queue, OPENGL_TO_WGPU_MATRIX * mat)?;
        Ok(())
    }
}
