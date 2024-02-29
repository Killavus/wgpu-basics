#import gpubasics::global::bindings::{camera, projection};
#import gpubasics::phong::bindings;
#import gpubasics::phong::vertex_output::VertexOutput;
#import gpubasics::phong::light_functions::fragmentLight;
#import gpubasics::instances::model::{Instance, model, model_invt};
#import gpubasics::vertex_data::Vertex;

@vertex
fn vs_main(v: Vertex, i: Instance) -> VertexOutput {
    var model = model(i);
    var inv_model_t = model_invt(i);

    var world_v = model * vec4<f32>(v.model_v, 1.0);
    var camera_v = camera * world_v;
    var ndc_v = projection * camera_v;

    var out: VertexOutput;
    out.position = ndc_v;
    out.w_pos = world_v;
    out.c_pos = camera_v;

    #ifndef VERTEX_PNTBUV
    out.normal = normalize(inv_model_t * vec4(v.normal_v, 0.0));
    #endif

    #ifdef VERTEX_PNTBUV
    // Since we are averaging tangents and bitangents for shared vertices
    // they can be not orthogonal to normal anymore.
    // We can apply Gram-Schmidt process to re-orthogonalize them.
    // This is happening here:
    out.t = normalize(inv_model_t * vec4(v.tangent_v, 0.0)).xyz;
    out.n = normalize(inv_model_t * vec4(v.normal_v, 0.0)).xyz;
    // re-orthogonalize t vector.
    out.t = normalize(out.t - dot(out.n, out.t) * out.n);
    out.b = cross(out.n, out.t);
    #endif

    #ifndef VERTEX_PN
    out.uv = v.uv;
    #endif

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = fragmentLight(in);

    return vec4(color, 1.0);
}
