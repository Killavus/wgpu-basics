

@group(0) @binding(0) var<uniform> camera: mat4x4<f32>;
@group(0) @binding(1) var<uniform> projection: mat4x4<f32>;
@group(0) @binding(2) var<uniform> inv_projection: mat4x4<f32>;
@group(0) @binding(3) var<uniform> inv_camera: mat4x4<f32>;
@group(0) @binding(4) var<uniform> model: mat4x4<f32>;
@group(0) @binding(5) var<uniform> inv_model_t: mat4x4<f32>;
@group(0) @binding(6) var<uniform> albedo: vec3<f32>;
@group(0) @binding(7) var<uniform> light: mat4x4<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec4<f32>,
    @location(1) w_pos: vec4<f32>,
};

// N * V = 0
// N * (M * V) = 0
@vertex
fn vs_main(@location(0) model_v: vec3<f32>, @location(1) normal_v: vec3<f32>) -> VertexOutput {
    var world_v = model * vec4<f32>(model_v, 1.0);
    var camera_v = projection * camera * world_v;

    var out: VertexOutput;
    out.position = camera_v;
    out.normal = normalize(inv_model_t * vec4<f32>(normal_v, 0.0));
    out.w_pos = world_v;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var lightColor = vec3<f32>(1.0, 1.0, 1.0);

    var ambientStrength = 0.1;
    var ambient = ambientStrength * lightColor;

    var lightPos = vec4(light[3].xyz, 0.0);

    var diffuseStrength = 0.7;
    var lightDir = normalize(lightPos - in.w_pos);
    var diffuseCoeff = max(dot(in.normal, lightDir), 0.0);
    var diffuse = diffuseStrength * diffuseCoeff * lightColor;

    var specularStrength = 0.2;
    var viewPos = vec4(inv_camera[3].xyz, 0.0);
    var viewDir = normalize(viewPos - in.w_pos);
    var reflectDir = reflect(-lightDir, in.normal);
    var specularCoeff = pow(max(dot(viewDir, reflectDir), 0.0), 256.0);
    var specular = specularStrength * specularCoeff * lightColor;

    return vec4(ambient + diffuse + specular, 1.0) * vec4(albedo, 1.0);
}
