// The idea of shadow mapping is just to generate a depth map from the light's perspective.
// That means we don't need the fragment pass, we just need a vertex pass to push our vertices
// to clip space.

@group(0) @binding(0) var<uniform> camera: mat4x4<f32>;
@group(0) @binding(1) var<uniform> projection: mat4x4<f32>;

struct VertexIn {
    @location(0) model_v: vec3<f32>,
    @location(1) normal_v: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct Instance {
    @location(3) model_c0: vec4<f32>,
    @location(4) model_c1: vec4<f32>,
    @location(5) model_c2: vec4<f32>,
    @location(6) model_c3: vec4<f32>,
    @location(7) model_invt_c0: vec4<f32>,
    @location(8) model_invt_c1: vec4<f32>,
    @location(9) model_invt_c2: vec4<f32>,
    @location(10) model_invt_c3: vec4<f32>,
}

@vertex
fn vs_main(v: VertexIn, i: Instance) -> @builtin(position) vec4<f32> {
    var model = mat4x4<f32>(i.model_c0, i.model_c1, i.model_c2, i.model_c3);

    var world_v = model * vec4<f32>(v.model_v, 1.0);
    var camera_v = projection * camera * world_v;

    return camera_v;
}
