#define_import_path gpubasics::forward::outputs::vertex

#ifdef VERTEX_PN
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec4<f32>,
    @location(1) w_pos: vec4<f32>,
    @location(2) c_pos: vec4<f32>,
};
#endif

#ifdef VERTEX_PNUV
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec4<f32>,
    @location(1) w_pos: vec4<f32>,
    @location(2) c_pos: vec4<f32>,
    @location(3) uv: vec2<f32>,
};
#endif

#ifdef VERTEX_PNTBUV
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) w_pos: vec4<f32>,
    @location(1) c_pos: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) t: vec3<f32>,
    @location(4) b: vec3<f32>,
    @location(5) n: vec3<f32>,
};
#endif

fn worldPos(in: VertexOutput) -> vec4<f32> {
    return in.w_pos;
}

fn cameraPos(in: VertexOutput) -> vec4<f32> {
    return in.c_pos;
}
