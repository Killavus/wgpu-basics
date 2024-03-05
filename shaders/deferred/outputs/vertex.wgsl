#define_import_path gpubasics::deferred::outputs::vertex

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) clip: vec4<f32>,
    @location(1) uv: vec2<f32>,
};
