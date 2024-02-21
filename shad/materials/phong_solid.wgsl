#define_import_path gpubasics::materials::phong_solid

struct PhongSolidMat {
    ambient: vec4<f32>,
    diffuse: vec4<f32>,
    specular: vec4<f32>,
}

@group(2) @binding(0) var<uniform> material: PhongSolidMat;
