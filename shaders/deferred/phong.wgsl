#import gpubasics::global::bindings::{camera_model, projection_invt};
#import gpubasics::phong::light_defs::{Light, Lights};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@group(1) @binding(0) var<storage, read> lights: Lights;
@group(1) @binding(1) var g_sampler: sampler;
@group(1) @binding(2) var g_normal: texture_2d<f32>;
@group(1) @binding(2) var g_diffuse: texture_2d<f32>;
@group(1) @binding(3) var g_specular: texture_2d<f32>;
@group(1) @binding(4) var g_depth: texture_depth_2d;

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    var VERTEX: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0)
    );

    var TEX: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0)
    );

    out.clip_position = vec4<f32>(VERTEX[in_vertex_index], 0.0, 1.0);
    out.tex_coords = vec2<f32>(TEX[in_vertex_index]);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4(1.0, 0.0, 0.0, 1.0);
}
