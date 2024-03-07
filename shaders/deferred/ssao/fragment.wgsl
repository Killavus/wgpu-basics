#define_import_path gpubasics::deferred::ssao::fragment
#import gpubasics::deferred::ssao::bindings::{g_sampler, g_normal, g_depth, noise_sampler, t_noise};
#import gpubasics::global::bindings::{camera_model, projection_invt};
#import gpubasics::deferred::outputs::vertex::VertexOutput;


fn depth(uv: vec2<f32>) -> f32 {
    return textureSample(g_depth, g_sampler, uv);
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

fn noise(in: VertexOutput) -> vec3<f32> {
    var noiseSize = textureDimensions(t_noise).xy;
    var viewSize = textureDimensions(g_normal).xy;

    var noiseScale = vec2<f32>(f32(viewSize.x) / f32(noiseSize.x), f32(viewSize.y) / f32(noiseSize.y));
    return textureSample(t_noise, noise_sampler, noiseScale * in.uv).rgb;
}
