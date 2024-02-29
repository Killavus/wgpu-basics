#define_import_path gpubasics::materials::phong_textured
#import gpubasics::phong::vertex_output::VertexOutput;

@group(2) @binding(0) var diffuse_t: texture_2d<f32>;
@group(2) @binding(1) var specular_t: texture_2d<f32>;
#ifdef NORMAL_MAP
@group(2) @binding(2) var normal_t: texture_2d<f32>;
@group(2) @binding(3) var mat_sampler: sampler;
@group(2) @binding(4) var<uniform> uShininess: f32;
#else
@group(2) @binding(2) var mat_sampler: sampler;
@group(2) @binding(3) var<uniform> uShininess: f32;
#endif

fn materialDiffuse(in: VertexOutput) -> vec3<f32> {
    return textureSample(diffuse_t, mat_sampler, in.uv).rgb;
}

fn materialSpecular(in: VertexOutput) -> vec3<f32> {
    return textureSample(specular_t, mat_sampler, in.uv).rgb;
}

fn materialAmbient(in: VertexOutput) -> vec3<f32> {
    return textureSample(diffuse_t, mat_sampler, in.uv).rgb;
}

fn shininess(in: VertexOutput) -> f32 {
    return uShininess;
}

#ifdef NORMAL_MAP
fn normal(in: VertexOutput) -> vec3<f32> {
    var tbn = mat3x3<f32>(in.t, in.b, in.n);
    return normalize(tbn * (textureSample(normal_t, mat_sampler, in.uv).rgb * 2.0 - 1.0));
}
#else
fn normal(in: VertexOutput) -> vec3<f32> {
    return in.normal.xyz;
}
#endif
