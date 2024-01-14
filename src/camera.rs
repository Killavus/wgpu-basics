use crate::gpu::GpuMat4;
use anyhow::Result;
use nalgebra as na;

#[derive(Clone, Copy)]
pub struct Camera {
    position: na::Point3<f32>,
    delta: na::Vector3<f32>,
    pitch: f32,
    yaw: f32,
}

impl Camera {
    pub fn new(position: na::Point3<f32>, pitch: f32, yaw: f32) -> Self {
        Self {
            position,
            delta: na::Vector3::zeros(),
            pitch,
            yaw,
        }
    }

    pub fn fly(&mut self, d: f32) {
        self.delta += na::Vector3::y() * d;
    }

    pub fn strafe(&mut self, d: f32) {
        let target = na::Vector3::new(
            self.pitch.cos() * self.yaw.cos(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.sin(),
        );

        let right = target.cross(&na::Vector3::y()).normalize();
        self.delta += right * d;
    }

    pub fn forwards(&mut self, d: f32) {
        let target = na::Vector3::new(
            self.pitch.cos() * self.yaw.cos(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.sin(),
        )
        .normalize();

        self.delta += target * d;
    }

    pub fn tilt_horizontally(&mut self, d: f32) {
        self.yaw += d;
    }

    pub fn tilt_vertically(&mut self, d: f32) {
        self.pitch += d;
    }

    pub fn target(&self) -> na::Point3<f32> {
        let target = na::Vector3::new(
            self.pitch.cos() * self.yaw.cos(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.sin(),
        );

        let position_now = self.position + self.delta;
        position_now + target
    }

    pub fn look_at_matrix(&self) -> na::Matrix4<f32> {
        let position_now = self.position + self.delta;

        na::Matrix4::look_at_rh(&position_now, &self.target(), &na::Vector3::y())
    }

    pub fn into_gpu(self, device: &wgpu::Device) -> GpuCamera {
        GpuCamera::new(self, device).unwrap()
    }
}

pub struct GpuCamera {
    camera: Camera,
    gpu_mat: GpuMat4,
    gpu_inv_mat: GpuMat4,
}

impl GpuCamera {
    pub fn new(camera: Camera, device: &wgpu::Device) -> Result<Self> {
        Ok(Self {
            camera,
            gpu_mat: GpuMat4::new(camera.look_at_matrix(), device)?,
            gpu_inv_mat: GpuMat4::new(camera.look_at_matrix().try_inverse().unwrap(), device)?,
        })
    }

    pub fn look_at_matrix(&self) -> na::Matrix4<f32> {
        self.camera.look_at_matrix()
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        self.gpu_mat.buffer()
    }

    pub fn model_buffer(&self) -> &wgpu::Buffer {
        self.gpu_inv_mat.buffer()
    }

    pub fn update<F>(&mut self, queue: &wgpu::Queue, updater: F) -> Result<()>
    where
        F: Fn(&mut Camera),
    {
        updater(&mut self.camera);

        self.gpu_mat.update(queue, self.camera.look_at_matrix())?;
        self.gpu_inv_mat
            .update(queue, self.camera.look_at_matrix().try_inverse().unwrap())?;
        Ok(())
    }
}
