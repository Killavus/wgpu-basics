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

pub fn wgpu_projection(proj_mat: na::Matrix4<f32>) -> na::Matrix4<f32> {
    OPENGL_TO_WGPU_MATRIX * proj_mat
}

pub struct GpuProjection(GpuMat4, GpuMat4);

impl GpuProjection {
    pub fn new(mat: na::Matrix4<f32>, device: &wgpu::Device) -> Result<Self> {
        let projection = OPENGL_TO_WGPU_MATRIX * mat;
        let projection_inv = projection
            .try_inverse()
            .ok_or_else(|| anyhow::anyhow!("failed to invert projection matrix"))?;

        Ok(Self(
            GpuMat4::new(projection, device)?,
            GpuMat4::new(projection_inv, device)?,
        ))
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        self.0.buffer()
    }

    pub fn inverse_buffer(&self) -> &wgpu::Buffer {
        self.1.buffer()
    }

    pub fn update(&mut self, queue: &wgpu::Queue, mat: na::Matrix4<f32>) -> Result<()> {
        let projection = OPENGL_TO_WGPU_MATRIX * mat;
        let projection_inv = projection
            .try_inverse()
            .ok_or_else(|| anyhow::anyhow!("failed to invert projection matrix"))?;

        self.0.update(queue, projection)?;
        self.1.update(queue, projection_inv)?;
        Ok(())
    }
}
