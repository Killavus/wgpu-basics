#define_import_path gpubasics::vertex_data

#ifdef VERTEX_PN
struct Vertex {
    @location(0) model_v: vec3<f32>,
    @location(1) normal_v: vec3<f32>,
};
#endif

#ifdef VERTEX_PNUV
struct Vertex {
    @location(0) model_v: vec3<f32>,
    @location(1) normal_v: vec3<f32>,
    @location(2) uv: vec2<f32>,
};
#endif

#ifdef VERTEX_PNTBUV
struct Vertex {
    @location(0) model_v: vec3<f32>,
    @location(1) normal_v: vec3<f32>,
    @location(2) tangent_v: vec3<f32>,
    @location(3) bitangent_v: vec3<f32>,
    @location(4) uv: vec2<f32>,
};
#endif
