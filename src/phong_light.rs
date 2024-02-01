use encase::{ArrayLength, ShaderType};
use nalgebra as na;

// We reuse w component of the structure, because:
// * According to Mario, GPU is aligning to vec4s anyway.
// * We can tightly pack the structure this way, avoiding unnecessary padding.
#[derive(ShaderType, Clone, Copy)]
pub struct PhongLight {
    // w = angle if light is spot light
    pub position: na::Vector4<f32>,
    // w = unused
    pub direction: na::Vector4<f32>,
    // w = k_c of attenuation
    pub ambient: na::Vector4<f32>,
    // w = k_l of attenuation
    pub diffuse: na::Vector4<f32>,
    // w = k_q of attenuation
    pub specular: na::Vector4<f32>,
}

#[derive(ShaderType)]
pub struct GpuPhongLights {
    num_directional: u32,
    num_point: u32,
    num_spot: u32,
    size: ArrayLength,
    #[size(runtime)]
    lights: Vec<PhongLight>,
}

#[derive(Default)]
pub struct PhongLightScene {
    pub directional: Vec<PhongLight>,
    pub point: Vec<PhongLight>,
    pub spot: Vec<PhongLight>,
}

impl PhongLightScene {
    pub fn new_point(
        &mut self,
        position: na::Vector3<f32>,
        ambient: na::Vector3<f32>,
        diffuse: na::Vector3<f32>,
        specular: na::Vector3<f32>,
        attenuation: na::Vector3<f32>,
    ) {
        self.point.push(PhongLight::new_point(
            position,
            ambient,
            diffuse,
            specular,
            attenuation,
        ));
    }

    pub fn new_directional(
        &mut self,
        direction: na::Vector3<f32>,
        ambient: na::Vector3<f32>,
        diffuse: na::Vector3<f32>,
        specular: na::Vector3<f32>,
    ) {
        self.directional.push(PhongLight::new_directional(
            direction, ambient, diffuse, specular,
        ));
    }

    pub fn new_spot(
        &mut self,
        position: na::Vector3<f32>,
        direction: na::Vector3<f32>,
        ambient: na::Vector3<f32>,
        diffuse: na::Vector3<f32>,
        specular: na::Vector3<f32>,
        angle: f32,
        attenuation: na::Vector3<f32>,
    ) {
        self.spot.push(PhongLight::new_spot(
            position,
            direction,
            ambient,
            diffuse,
            specular,
            angle,
            attenuation,
        ));
    }

    pub fn into_gpu(&self) -> GpuPhongLights {
        GpuPhongLights {
            num_directional: self.directional.len() as u32,
            num_point: self.point.len() as u32,
            num_spot: self.spot.len() as u32,
            size: ArrayLength,
            lights: self
                .directional
                .iter()
                .copied()
                .chain(self.point.iter().copied())
                .chain(self.spot.iter().copied())
                .collect(),
        }
    }
}

impl PhongLight {
    pub fn new_point(
        position: na::Vector3<f32>,
        ambient: na::Vector3<f32>,
        diffuse: na::Vector3<f32>,
        specular: na::Vector3<f32>,
        attenuation: na::Vector3<f32>,
    ) -> Self {
        Self {
            position: na::Vector4::new(position.x, position.y, position.z, 0.0),
            direction: na::Vector4::zeros(),
            ambient: na::Vector4::new(ambient.x, ambient.y, ambient.z, attenuation.x),
            diffuse: na::Vector4::new(diffuse.x, diffuse.y, diffuse.z, attenuation.y),
            specular: na::Vector4::new(specular.x, specular.y, specular.z, attenuation.z),
        }
    }

    pub fn new_directional(
        direction: na::Vector3<f32>,
        ambient: na::Vector3<f32>,
        diffuse: na::Vector3<f32>,
        specular: na::Vector3<f32>,
    ) -> Self {
        Self {
            position: na::Vector4::zeros(),
            direction: na::Vector4::new(direction.x, direction.y, direction.z, 0.0),
            ambient: na::Vector4::new(ambient.x, ambient.y, ambient.z, 0.0),
            diffuse: na::Vector4::new(diffuse.x, diffuse.y, diffuse.z, 0.0),
            specular: na::Vector4::new(specular.x, specular.y, specular.z, 0.0),
        }
    }

    pub fn new_spot(
        position: na::Vector3<f32>,
        direction: na::Vector3<f32>,
        ambient: na::Vector3<f32>,
        diffuse: na::Vector3<f32>,
        specular: na::Vector3<f32>,
        angle: f32,
        attenuation: na::Vector3<f32>,
    ) -> Self {
        Self {
            position: na::Vector4::new(position.x, position.y, position.z, angle),
            direction: na::Vector4::new(direction.x, direction.y, direction.z, 0.0),
            ambient: na::Vector4::new(ambient.x, ambient.y, ambient.z, attenuation.x),
            diffuse: na::Vector4::new(diffuse.x, diffuse.y, diffuse.z, attenuation.y),
            specular: na::Vector4::new(specular.x, specular.y, specular.z, attenuation.z),
        }
    }
}
