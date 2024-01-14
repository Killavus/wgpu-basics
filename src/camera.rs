use anyhow::Result;
use encase::{ShaderSize, UniformBuffer};
use nalgebra as na;

const SIZE: u64 = na::Matrix4::<f32>::SHADER_SIZE.into();

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
    camera_buf: wgpu::Buffer,
}

impl GpuCamera {
    pub fn new(camera: Camera, device: &wgpu::Device) -> Result<Self> {
        use wgpu::util::DeviceExt;

        let mut contents = UniformBuffer::new(Vec::with_capacity(SIZE as usize));
        contents.write(&camera.look_at_matrix())?;

        let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: contents.into_inner().as_slice(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Ok(Self { camera, camera_buf })
    }

    pub fn look_at_matrix(&self) -> na::Matrix4<f32> {
        self.camera.look_at_matrix()
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.camera_buf
    }

    pub fn update<F>(&mut self, queue: &wgpu::Queue, updater: F) -> Result<()>
    where
        F: Fn(&mut Camera),
    {
        updater(&mut self.camera);

        let mut contents = UniformBuffer::new(Vec::with_capacity(SIZE as usize));
        contents.write(&self.camera.look_at_matrix())?;
        queue.write_buffer(&self.camera_buf, 0, contents.into_inner().as_slice());
        Ok(())
    }
}
