struct Light {
    light_type: u32,
    position: vec3<f32>,
    direction: vec3<f32>,
    color: vec3<f32>,
    angle: f32,
    casting_shadows: u32,
    attenuation: vec3<f32>,
};

struct Lights {
    length: u32,
    lights: array<Light>,
};

const LIGHT_POINT: u32 = u32(0);
const LIGHT_DIRECTIONAL: u32 = u32(1);
const LIGHT_SPOT: u32 = u32(2);

@group(0) @binding(0) var<uniform> camera: mat4x4<f32>;
@group(0) @binding(1) var<uniform> camera_model: mat4x4<f32>;
@group(0) @binding(2) var<uniform> projection: mat4x4<f32>;
@group(0) @binding(3) var<storage, read> lights: Lights;

struct VertexIn {
    @location(0) model_v: vec3<f32>,
    @location(1) normal_v: vec3<f32>,
};

struct Instance {
    @location(2) model_c0: vec4<f32>,
    @location(3) model_c1: vec4<f32>,
    @location(4) model_c2: vec4<f32>,
    @location(5) model_c3: vec4<f32>,
    @location(6) model_invt_c0: vec4<f32>,
    @location(7) model_invt_c1: vec4<f32>,
    @location(8) model_invt_c2: vec4<f32>,
    @location(9) model_invt_c3: vec4<f32>,
    @location(10) albedo: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec4<f32>,
    @location(1) w_pos: vec4<f32>,
    @location(2) c_pos: vec4<f32>,
    @location(3) albedo: vec3<f32>,
};

struct PhongSettings {
    ambientStrength: f32,
    diffuseStrength: f32,
    specularStrength: f32,
    specularCoeff: f32,
};

@group(1) @binding(0) var<uniform> settings: PhongSettings;

struct ShadowMapResult {
    num_splits: u32,
    split_depths: array<vec4<f32>, 16>
};

// Set to 12, because it has data at every 4th element.
// min_ubo_binding_size is 256, and mat4x4 is 64.
// so first array is on offset 0, then offset 256 and so on.
@group(2) @binding(0) var<uniform> light_view: array<mat4x4<f32>, 12>;
@group(2) @binding(1) var<uniform> light_projection: array<mat4x4<f32>, 12>;
@group(2) @binding(2) var smap_sampler: sampler;
@group(2) @binding(3) var smap: texture_depth_2d_array;
@group(2) @binding(4) var<uniform> smap_result: ShadowMapResult;

@vertex
fn vs_main(v: VertexIn, i: Instance) -> VertexOutput {
    var model = mat4x4<f32>(i.model_c0, i.model_c1, i.model_c2, i.model_c3);
    var inv_model_t = mat4x4<f32>(i.model_invt_c0, i.model_invt_c1, i.model_invt_c2, i.model_invt_c3);

    var world_v = model * vec4<f32>(v.model_v, 1.0);
    var camera_v = camera * world_v;
    var ndc_v = projection * camera_v;

    var out: VertexOutput;
    out.position = ndc_v;
    out.normal = normalize(inv_model_t * vec4(v.normal_v, 0.0));
    out.w_pos = world_v;
    out.c_pos = camera_v;
    out.albedo = i.albedo;

    return out;
}


fn calculateLight(in: VertexOutput, light: Light) -> vec3<f32> {
    var ambientStrength = settings.ambientStrength;
    var diffuseStrength = settings.diffuseStrength;
    var specularStrength = settings.specularStrength;
    var viewPos = camera_model[3].xyz;

    var lightDir = vec3(0.0, 0.0, 0.0);
    var color = vec3<f32>(0.0, 0.0, 0.0);

    var attenuation = 1.0;
    var lightDistance = 0.0;

    if light.light_type == LIGHT_DIRECTIONAL {
        lightDir = -light.direction;
    } else if light.light_type == LIGHT_POINT || light.light_type == LIGHT_SPOT {
        lightDir = normalize(light.position - in.w_pos.xyz);
        lightDistance = length(light.position - in.w_pos.xyz);

        attenuation = 1.0 / (light.attenuation.x + light.attenuation.y * lightDistance + light.attenuation.z * lightDistance * lightDistance);
    } else {
        return vec3<f32>(0.0, 0.0, 0.0);
    }

    color += attenuation * ambientStrength * light.color;

    var shadow = 0.0;
    if light.light_type == LIGHT_DIRECTIONAL {
        var split = -1;
        for (var i = 0; i < i32(smap_result.num_splits); i += 1) {
            if abs(in.c_pos.z) < smap_result.split_depths[i].x {
                split = i;
                break;
            }
        }

        if split > -1 {
            var l_pos = light_projection[split * 4] * light_view[split * 4] * in.w_pos;
            var lightPos = (l_pos.xyz / l_pos.w);
            var lightDepth = lightPos.z;

            var texSize = textureDimensions(smap).xy;

            var texelSize = vec2(1.0 / f32(texSize.x), 1.0 / f32(texSize.y));

            var bias = max(0.01 * (1.0 - dot(in.normal.xyz, lightDir)), 0.001);
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
    }

    if light.light_type == LIGHT_SPOT {
        // This is a cosine between lightDir and spotDir.
        var theta = dot(lightDir, normalize(-light.direction));
        var epsilon = cos(light.angle);

        if theta <= epsilon {
            return color;
        }
    }

    var diffuseCoeff = max(dot(in.normal.xyz, lightDir), 0.0);
    color += (1.0 - shadow) * attenuation * diffuseCoeff * light.color;
    var viewDir = normalize(viewPos - in.w_pos.xyz);
    var reflectDir = reflect(-lightDir, in.normal.xyz);
    var specularCoeff = pow(max(dot(viewDir, reflectDir), 0.0), settings.specularCoeff);
    color += (1.0 - shadow) * attenuation * specularStrength * specularCoeff * light.color;

    return color;
}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var lightColor = vec3<f32>(1.0, 1.0, 1.0);
    var color = vec3(0.0, 0.0, 0.0);

    for (var i = 0; u32(i) < lights.length; i = i + 1) {
        color += calculateLight(in, lights.lights[i]);
    }

    return vec4(color, 1.0) * vec4(in.albedo, 1.0);
}
