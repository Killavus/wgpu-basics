#define_import_path gpubasics::deferred::functions
#import gpubasics::deferred::bindings::{g_sampler, g_normal, g_diffuse, g_specular, g_depth};
#import gpubasics::deferred::vertex_output::VertexOutput;
#import gpubasics::global::bindings::{camera_model, projection_invt};

fn worldPos(in: VertexOutput) -> vec4<f32> {
    var depth = textureSample(g_depth, g_sampler, in.uv);
    var ndc = vec4<f32>(in.clip.x, in.clip.y, depth, 1.0);
    var clip = projection_invt * ndc;
    clip /= clip.w;

    return camera_model * clip;
}

fn cameraPos(in: VertexOutput) -> vec4<f32> {
    var depth = textureSample(g_depth, g_sampler, in.uv);
    var ndc = vec4<f32>(in.clip.x, in.clip.y, depth, 1.0);
    var clip = projection_invt * ndc;
    clip /= clip.w;

    return clip;
}

fn normal(in: VertexOutput) -> vec3<f32> {
    return textureSample(g_normal, g_sampler, in.uv).rgb;
}

fn ambient(in: VertexOutput) -> vec3<f32> {
    return textureSample(g_diffuse, g_sampler, in.uv).rgb;
}

fn diffuse(in: VertexOutput) -> vec3<f32> {
    return textureSample(g_diffuse, g_sampler, in.uv).rgb;
}

fn specular(in: VertexOutput) -> vec3<f32> {
    return textureSample(g_specular, g_sampler, in.uv).rgb;
}

fn shininess(in: VertexOutput) -> f32 {
    return textureSample(g_specular, g_sampler, in.uv).a * 256.0;
}
