use nalgebra as na;

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
}
