use encase::{ArrayLength, ShaderType};
use nalgebra as na;

#[derive(ShaderType, Clone, Copy)]
pub struct Light {
    light_type: u32,
    position_direction: na::Vector3<f32>,
    color: na::Vector3<f32>,
    angle: f32,
    casting_shadows: u32,
}

#[derive(ShaderType)]
pub struct GpuLights {
    pub size: ArrayLength,
    #[size(runtime)]
    pub lights: Vec<Light>,
}

impl Light {
    pub fn new_point(position: na::Vector3<f32>, color: na::Vector3<f32>) -> Self {
        Self {
            light_type: 0,
            position_direction: position,
            color,
            angle: 0.0,
            casting_shadows: 0,
        }
    }

    pub fn new_directional(direction: na::Vector3<f32>, color: na::Vector3<f32>) -> Self {
        Self {
            light_type: 1,
            position_direction: direction,
            color,
            angle: 0.0,
            casting_shadows: 0,
        }
    }

    pub fn new_spot(
        position: na::Vector3<f32>,
        direction: na::Vector3<f32>,
        color: na::Vector3<f32>,
        angle: f32,
    ) -> Self {
        Self {
            light_type: 2,
            position_direction: position,
            color,
            angle,
            casting_shadows: 0,
        }
    }

    pub fn toggle_shadow_casting(&mut self) {
        self.casting_shadows = 1;
    }
}
