#define_import_path gpubasics::shadow::cascaded::definitions

struct ShadowMapResult {
    num_splits: u32,
    split_depths: array<vec4<f32>, 16>
};

struct ShadowMapMatrices {
    cam_split_a: mat4x4<f32>,
    cam_split_b: mat4x4<f32>,
    cam_split_c: mat4x4<f32>,
    proj_split_a: mat4x4<f32>,
    proj_split_b: mat4x4<f32>,
    proj_split_c: mat4x4<f32>,
};
