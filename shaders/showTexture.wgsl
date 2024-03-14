#ifdef DEPTH_TEXTURE
@group(0) @binding(0) var texture: texture_depth_2d;
#else
@group(0) @binding(0) var texture: texture_2d<f32>;
#endif
@group(0) @binding(1) var t_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

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
    #ifdef DEPTH_TEXTURE
    var depth = textureSample(texture, t_sampler, in.tex_coords);
    return vec4(depth, depth, depth, 1.0);
    #else
    return textureSample(texture, t_sampler, in.tex_coords);
    #endif
}
