use encase::{ArrayLength, ShaderType};
use nalgebra as na;

pub const LIGHT_TYPE_POINT: u32 = 0;
pub const LIGHT_TYPE_DIRECTIONAL: u32 = 1;
pub const LIGHT_TYPE_SPOT: u32 = 2;

#[derive(ShaderType, Clone, Copy)]
pub struct Light {
    pub light_type: u32,
    pub position: na::Vector3<f32>,
    pub direction: na::Vector3<f32>,
    pub color: na::Vector3<f32>,
    pub angle: f32,
    pub casting_shadows: u32,
    pub attenuation: na::Vector3<f32>,
}

#[derive(ShaderType)]
pub struct GpuLights {
    pub size: ArrayLength,
    #[size(runtime)]
    pub lights: Vec<Light>,
}

impl Light {
    pub fn new_point(
        position: na::Vector3<f32>,
        color: na::Vector3<f32>,
        attenuation: na::Vector3<f32>,
    ) -> Self {
        Self {
            light_type: LIGHT_TYPE_POINT,
            position,
            direction: na::Vector3::zeros(),
            color,
            angle: 0.0,
            casting_shadows: 0,
            attenuation,
        }
    }

    pub fn new_directional(direction: na::Vector3<f32>, color: na::Vector3<f32>) -> Self {
        Self {
            light_type: LIGHT_TYPE_DIRECTIONAL,
            position: na::Vector3::zeros(),
            direction,
            color,
            angle: 0.0,
            casting_shadows: 0,
            attenuation: na::Vector3::zeros(),
        }
    }

    pub fn new_spot(
        position: na::Vector3<f32>,
        direction: na::Vector3<f32>,
        color: na::Vector3<f32>,
        angle: f32,
        attenuation: na::Vector3<f32>,
    ) -> Self {
        Self {
            light_type: LIGHT_TYPE_SPOT,
            position,
            direction,
            color,
            angle,
            casting_shadows: 0,
            attenuation,
        }
    }

    pub fn toggle_shadow_casting(&mut self) {
        self.casting_shadows = 1;
    }
}
