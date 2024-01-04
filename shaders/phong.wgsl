

@group(0) @binding(0) var<uniform> camera: mat4x4<f32>;
@group(0) @binding(1) var<uniform> model: mat4x4<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec4<f32>,
};


@vertex
fn vs_main(@location(0) model_v: vec3<f32>, @location(1) normal_v: vec3<f32>, @builtin(vertex_index) idx: u32) -> VertexOutput {
    var world_v = model * vec4<f32>(model_v, 1.0);
    var camera_v = camera * world_v;

    var out: VertexOutput;
    out.position = camera_v;
    out.normal = vec4<f32>(normal_v, 1.0);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = (in.normal + 1.0) * 0.5;

    return color;
}
