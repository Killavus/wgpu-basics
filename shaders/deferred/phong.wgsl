#import gpubasics::deferred::shaders::screen_quad_vs::screenQuad;
#import gpubasics::deferred::outputs::vertex::VertexOutput;
#import gpubasics::phong::functions::fragmentLight;

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    return screenQuad(in_vertex_index);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = fragmentLight(in);

    return vec4(color, 1.0);
}
