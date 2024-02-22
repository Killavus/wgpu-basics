struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

// 1. Fetch actual depth information from the depth buffer.
// 2. Sample N points in the hemisphere near the fragment. If z value is deeper than fragment sample, append occlusion.
// 3. Calculate occlusion factor by sampling all N points and dividing by N.
// 4. We need: precalculated random points & noise texture for random rotations.
// 5. In order to do random rotations we just need to create a TBN matrix from random tangent vector by sampling the noise texture.

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

