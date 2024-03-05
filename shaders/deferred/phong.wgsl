#import gpubasics::deferred::vertex_output::VertexOutput;
#import gpubasics::deferred::functions::worldPos;
#import gpubasics::phong::light_functions::fragmentLight;
#import gpubasics::global::bindings::{projection, camera};

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

    out.position = vec4<f32>(VERTEX[in_vertex_index], 0.0, 1.0);
    out.clip = vec4<f32>(VERTEX[in_vertex_index], 0.0, 1.0);
    out.uv = vec2<f32>(TEX[in_vertex_index]);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = fragmentLight(in);

    return vec4(color, 1.0);
}
