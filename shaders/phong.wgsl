

@group(0) @binding(0) var<uniform> camera: mat4x4<f32>;
@group(0) @binding(1) var<uniform> projection: mat4x4<f32>;
@group(0) @binding(2) var<uniform> inv_projection: mat4x4<f32>;
@group(0) @binding(3) var<uniform> inv_camera: mat4x4<f32>;
@group(0) @binding(4) var<uniform> model: mat4x4<f32>;
@group(0) @binding(5) var<uniform> inv_model_t: mat4x4<f32>;
@group(0) @binding(6) var<uniform> albedo: vec3<f32>;

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
    out.normal = inv_model_t * vec4<f32>(normal_v, 0.0);
    out.w_pos = world_v;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = vec4<f32>(albedo, 1.0);

    var lightPos = inv_camera[3];
    var lightDir = normalize(lightPos - in.w_pos);
    var lightColor = vec3<f32>(1.0, 1.0, 1.0);
    var lightIntensity = 5.0;
    var lightDistance = length(lightPos - in.w_pos);
    var attenuation = 1.0 / (lightDistance * lightDistance);
    var diffuse = max(dot(in.normal, lightDir), 0.0);


    return 0.1 * vec4(albedo, 1.0) + 0.9 * diffuse * lightIntensity * attenuation * vec4<f32>(lightColor, 1.0);
}
