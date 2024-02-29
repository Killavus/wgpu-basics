#define_import_path gpubasics::phong::light_defs

struct Light {
    position: vec4<f32>,
    direction: vec4<f32>,
    ambient: vec4<f32>,
    diffuse: vec4<f32>,
    specular: vec4<f32>,
};

struct Lights {
    num_directional: u32,
    num_point: u32,
    num_spot: u32,
    length: u32,
    lights: array<Light>,
};
