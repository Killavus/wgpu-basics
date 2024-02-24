#define_import_path gpubasics::materials::phong_solid
#import gpubasics::phong::vertex_output::VertexOutput;

#ifdef NORMAL_MAP
@group(2) @binding(2) var normal_t: texture_2d<f32>;
@group(2) @binding(3) var mat_sampler: sampler;
#endif

struct PhongSolidMat {
    ambient: vec4<f32>,
    diffuse: vec4<f32>,
    specular: vec4<f32>,
}

@group(2) @binding(0) var<uniform> material: PhongSolidMat;

fn materialDiffuse(in: VertexOutput) -> vec3<f32> {
    return material.diffuse.xyz;
}

fn materialSpecular(in: VertexOutput) -> vec3<f32> {
    return material.specular.xyz;
}

fn materialAmbient(in: VertexOutput) -> vec3<f32> {
    return material.ambient.xyz;
}

fn shininess(in: VertexOutput) -> f32 {
    return material.specular.w;
}

#ifdef NORMAL_MAP
fn normal(in: VertexOutput) -> vec3<f32> {
    var tbn = mat3x3<f32>(in.t, in.b, in.normal);
    return normalize(tbn * (textureSample(normal_t, mat_sampler, in.uv).rgb * 2.0 - 1.0));
}
#else
fn normal(in: VertexOutput) -> vec3<f32> {
    return in.normal.xyz;
}
#endif
