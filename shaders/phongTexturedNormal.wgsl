const LIGHT_POINT: u32 = u32(0);
const LIGHT_DIRECTIONAL: u32 = u32(1);
const LIGHT_SPOT: u32 = u32(2);

struct Light {
    position: vec4<f32>,
    direction: vec4<f32>,
    ambient: vec4<f32>,
    diffuse: vec4<f32>,
    specular: vec4<f32>,
};

struct Lights {
    num_directional: u32,
    num_point: u32,
    num_spot: u32,
    length: u32,
    lights: array<Light>,
};

struct ShadowMapResult {
    num_splits: u32,
    split_depths: array<vec4<f32>, 16>
};


struct ShadowMapMatrices {
    cam_split_1: mat4x4<f32>,
    cam_split_2: mat4x4<f32>,
    cam_split_3: mat4x4<f32>,
    proj_split_1: mat4x4<f32>,
    proj_split_2: mat4x4<f32>,
    proj_split_3: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: mat4x4<f32>;
@group(0) @binding(1) var<uniform> projection: mat4x4<f32>;
@group(0) @binding(2) var<uniform> camera_model: mat4x4<f32>;

@group(1) @binding(0) var<storage, read> lights: Lights;

@group(2) @binding(0) var diffuse_t: texture_2d<f32>;
@group(2) @binding(1) var specular_t: texture_2d<f32>;
@group(2) @binding(2) var normal_t: texture_2d<f32>;
@group(2) @binding(3) var mat_sampler: sampler;

@group(3) @binding(0) var<uniform> smap_matrices: ShadowMapMatrices;
@group(3) @binding(1) var smap_sampler: sampler;
@group(3) @binding(2) var smap: texture_depth_2d_array;
@group(3) @binding(3) var<uniform> smap_result: ShadowMapResult;

struct VertexIn {
    @location(0) model_v: vec3<f32>,
    @location(1) normal_v: vec3<f32>,
    @location(2) tangent_v: vec3<f32>,
    @location(3) bitangent_v: vec3<f32>,
    @location(4) uv: vec2<f32>,
};

struct Instance {
    @location(5) model_c0: vec4<f32>,
    @location(6) model_c1: vec4<f32>,
    @location(7) model_c2: vec4<f32>,
    @location(8) model_c3: vec4<f32>,
    @location(9) model_invt_c0: vec4<f32>,
    @location(10) model_invt_c1: vec4<f32>,
    @location(11) model_invt_c2: vec4<f32>,
    @location(12) model_invt_c3: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) w_pos: vec4<f32>,
    @location(1) c_pos: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) t: vec3<f32>,
    @location(4) b: vec3<f32>,
    @location(5) n: vec3<f32>,
};

@vertex
fn vs_main(v: VertexIn, i: Instance) -> VertexOutput {
    var model = mat4x4<f32>(i.model_c0, i.model_c1, i.model_c2, i.model_c3);
    var inv_model_t = mat4x4<f32>(i.model_invt_c0, i.model_invt_c1, i.model_invt_c2, i.model_invt_c3);

    var world_v = model * vec4<f32>(v.model_v, 1.0);
    var camera_v = camera * world_v;
    var ndc_v = projection * camera_v;

    var out: VertexOutput;
    out.position = ndc_v;
    // Since we are averaging tangents and bitangents for shared vertices
    // they can be not orthogonal to normal anymore.
    // We can apply Gram-Schmidt process to re-orthogonalize them.
    // This is happening here:
    out.t = normalize(inv_model_t * vec4(v.tangent_v, 0.0)).xyz;
    out.n = normalize(inv_model_t * vec4(v.normal_v, 0.0)).xyz;
    // re-orthogonalize t vector.
    out.t = normalize(out.t - dot(out.n, out.t) * out.n);
    out.b = cross(out.n, out.t);
    out.w_pos = world_v;
    out.c_pos = camera_v;
    out.uv = v.uv;

    return out;
}

fn calculateShadow(in: VertexOutput, normal: vec3<f32>, lightDir: vec3<f32>) -> f32 {
    var shadow = 0.0;
    var split = -1;
    var light_cam_mats = array<mat4x4<f32>, 3>(smap_matrices.cam_split_1, smap_matrices.cam_split_2, smap_matrices.cam_split_3);
    var light_proj_mats = array<mat4x4<f32>, 3>(smap_matrices.proj_split_1, smap_matrices.proj_split_2, smap_matrices.proj_split_3);

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


fn calculateLight(in: VertexOutput, light: Light, light_type: u32) -> vec3<f32> {
    var lightAmbient = light.ambient.xyz;
    var lightDiffuse = light.diffuse.xyz;
    var lightSpecular = light.specular.xyz;

    var attenuationConstant = light.ambient.w;
    var attenuationLinear = light.diffuse.w;
    var attenuationQuadratic = light.specular.w;

    var lightPosition = light.position.xyz;
    var lightDirection = light.direction.xyz;

    var shininess = 32.0;

    var viewPos = camera_model[3].xyz;

    var lightDir = vec3(0.0, 0.0, 0.0);
    var color = vec3<f32>(0.0, 0.0, 0.0);

    var attenuation = 1.0;
    var lightDistance = 0.0;

    // Here we transform tangent-space vector into world space.
    // We can do other way around - transform lightDir & position to tangent space.
    // This is potentially better for performance (we can do it in vertex shader),
    // but don't know if it works with shadow maps and other stuff.
    // TBN is tangent space -> world space.
    // To do world space -> tangent space we need to transpose TBN.
    // Transpose works because it is an orthogonal matrix.

    var tbn = mat3x3<f32>(in.t, in.b, in.n);
    var normal = normalize(tbn * (textureSample(normal_t, mat_sampler, in.uv).rgb * 2.0 - 1.0));

    var shadow = 0.0;
    if light_type == LIGHT_DIRECTIONAL {
        lightDir = -lightDirection;
        shadow = calculateShadow(in, normal, lightDir);
    } else if light_type == LIGHT_POINT || light_type == LIGHT_SPOT {
        lightDir = normalize(lightPosition - in.w_pos.xyz);
        lightDistance = length(lightPosition - in.w_pos.xyz);

        attenuation = 1.0 / (attenuationConstant + attenuationLinear * lightDistance + attenuationQuadratic * lightDistance * lightDistance);
    } else {
        return vec3<f32>(0.0, 0.0, 0.0);
    }

    color += lightAmbient * textureSample(diffuse_t, mat_sampler, in.uv).rgb;

    if light_type == LIGHT_SPOT {
        // This is a cosine between lightDir and spotDir.
        var theta = dot(lightDir, normalize(-lightDirection));
        var angle = light.position.w;
        var epsilon = cos(angle);

        if theta <= epsilon {
            return color;
        }
    }

    var diffuseCoeff = max(dot(normal, lightDir), 0.0);
    color += textureSample(diffuse_t, mat_sampler, in.uv).rgb * ((1.0 - shadow) * attenuation * diffuseCoeff * lightDiffuse);
    var viewDir = normalize(viewPos - in.w_pos.xyz);
    var reflectDir = reflect(-lightDir, normal);
    var specularCoeff = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    color += textureSample(specular_t, mat_sampler, in.uv).rgb * ((1.0 - shadow) * attenuation * specularCoeff * lightSpecular);

    return color;
}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var lightColor = vec3<f32>(1.0, 1.0, 1.0);
    var color = vec3(0.0, 0.0, 0.0);

    for (var i = 0; u32(i) < lights.num_directional; i = i + 1) {
        color += calculateLight(in, lights.lights[i], LIGHT_DIRECTIONAL);
    }

    for (var i = u32(0); i < lights.num_point; i = i + 1) {
        color += calculateLight(in, lights.lights[i + lights.num_directional], LIGHT_POINT);
    }

    for (var i = u32(0); i < lights.num_spot; i = i + 1) {
        color += calculateLight(in, lights.lights[i + lights.num_directional + lights.num_point], LIGHT_SPOT);
    }

    color /= f32(lights.length);

    return vec4(color, 1.0);
}
