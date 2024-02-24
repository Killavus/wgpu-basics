#define_import_path gpubasics::csm::phong
#import gpubasics::phong::vertex_output::VertexOutput;

#ifdef MATERIAL_PHONG_SOLID
#import gpubasics::materials::phong_solid::{normal};
#endif

#ifdef MATERIAL_PHONG_TEXTURED
#import gpubasics::materials::phong_textured::{normal};
#endif


struct ShadowMapResult {
    num_splits: u32,
    split_depths: array<vec4<f32>, 16>
};

struct ShadowMapMatrices {
    cam_split_a: mat4x4<f32>,
    cam_split_b: mat4x4<f32>,
    cam_split_c: mat4x4<f32>,
    proj_split_a: mat4x4<f32>,
    proj_split_b: mat4x4<f32>,
    proj_split_c: mat4x4<f32>,
};

@group(3) @binding(0) var<uniform> smap_matrices: ShadowMapMatrices;
@group(3) @binding(1) var smap_sampler: sampler;
@group(3) @binding(2) var smap: texture_depth_2d_array;
@group(3) @binding(3) var<uniform> smap_result: ShadowMapResult;

fn calculateShadow(in: VertexOutput, lightDir: vec3<f32>) -> f32 {
    var shadow = 0.0;
    var split = -1;
    var light_cam_mats = array<mat4x4<f32>, 3>(smap_matrices.cam_split_a, smap_matrices.cam_split_b, smap_matrices.cam_split_c);
    var light_proj_mats = array<mat4x4<f32>, 3>(smap_matrices.proj_split_a, smap_matrices.proj_split_b, smap_matrices.proj_split_c);

    for (var i = 0; i < i32(smap_result.num_splits); i += 1) {
        if abs(in.c_pos.z) < smap_result.split_depths[i].x {
            split = i;
                break;
        }
    }

    if split > -1 {
        var l_pos = light_proj_mats[split] * light_cam_mats[split] * in.w_pos;
        var lightPos = (l_pos.xyz / l_pos.w);
        var lightDepth = lightPos.z;

        var normal = normal(in);

        var texSize = textureDimensions(smap).xy;
        var texelSize = vec2(1.0 / f32(texSize.x), 1.0 / f32(texSize.y));
        var bias = max(0.01 * (1.0 - dot(normal, lightDir)), 0.001);
        var texelPos = lightPos.xy;

        // Percentage Closer Filtering with 3x3.
        for (var x = -1; x <= 1; x += 1) {
            for (var y = -1; y <= 1; y += 1) {
                var shadowDepth = textureSample(smap, smap_sampler, (texelPos + vec2(f32(x), f32(y)) * texelSize) * vec2(0.5, -0.5) + 0.5, split);
                if (lightDepth - bias) > shadowDepth {
                    shadow += 1.0;
                }
            }
        }
        shadow /= 9.0;

        if lightDepth > 1.0 {
            shadow = 0.0;
        }
    }

    return shadow;
}
