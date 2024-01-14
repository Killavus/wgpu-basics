
@group(0) @binding(0) var<uniform> camera: mat4x4<f32>;
@group(0) @binding(1) var<uniform> projection: mat4x4<f32>;
@group(0) @binding(2) var<uniform> inv_projection: mat4x4<f32>;
@group(0) @binding(3) var<uniform> inv_camera: mat4x4<f32>;
@group(0) @binding(4) var<uniform> light: mat4x4<f32>;
@group(0) @binding(5) var shadowMap: texture_depth_2d;
@group(0) @binding(6) var shadowMapSampler: sampler;
@group(0) @binding(7) var<uniform> lightCamera: mat4x4<f32>;
@group(0) @binding(8) var<uniform> lightProjection: mat4x4<f32>;
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
    @location(2) l_pos: vec4<f32>,
    @location(3) albedo: vec3<f32>,
};

@vertex
fn vs_main(v: VertexIn, i: Instance) -> VertexOutput {
    var model = mat4x4<f32>(i.model_c0, i.model_c1, i.model_c2, i.model_c3);
    var inv_model_t = mat4x4<f32>(i.model_invt_c0, i.model_invt_c1, i.model_invt_c2, i.model_invt_c3);

    var world_v = model * vec4<f32>(v.model_v, 1.0);
    var camera_v = projection * camera * world_v;
    var light_v = lightProjection * lightCamera * world_v;

    var out: VertexOutput;
    out.position = camera_v;
    out.normal = normalize(inv_model_t * vec4(v.normal_v, 0.0));
    out.w_pos = world_v;
    out.l_pos = light_v;
    out.albedo = i.albedo;

    return out;
}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var lightColor = vec3<f32>(1.0, 1.0, 1.0);

    var ambientStrength = 0.1;
    var ambient = ambientStrength * lightColor;

    var lightPos = vec4(light[3].xyz, 0.0);

    var diffuseStrength = 0.5;
    var lightDir = normalize(lightPos - in.w_pos);
    var diffuseCoeff = max(dot(in.normal, lightDir), 0.0);
    var diffuse = diffuseStrength * diffuseCoeff * lightColor;

    var lPosDivided = in.l_pos.xyz / in.l_pos.w;
    var smapCoords = lPosDivided.xy * 0.5 + 0.5;
    var closestDepth = textureSample(shadowMap, shadowMapSampler, smapCoords);
    var currentDepth = in.l_pos.z / in.l_pos.w;

    var shadow = 0.0;
    if lPosDivided.z <= 1.0 && currentDepth > closestDepth {
        shadow = 1.0;
    }


    var specularStrength = 0.4;
    var viewPos = vec4(inv_camera[3].xyz, 0.0);
    var viewDir = normalize(viewPos - in.w_pos);
    var reflectDir = reflect(-lightDir, in.normal);
    var specularCoeff = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    var specular = specularStrength * specularCoeff * lightColor;

    return vec4(ambient + (1.0 - shadow) * (diffuse + specular), 1.0) * vec4(in.albedo, 1.0);
}
