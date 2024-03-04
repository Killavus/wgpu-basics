#define_import_path gpubasics::global::bindings

@group(0) @binding(0) var<uniform> camera: mat4x4<f32>;
@group(0) @binding(1) var<uniform> projection: mat4x4<f32>;
@group(0) @binding(2) var<uniform> camera_model: mat4x4<f32>;
@group(0) @binding(3) var<uniform> projection_invt: mat4x4<f32>;
