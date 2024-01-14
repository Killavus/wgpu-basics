struct Light {
    light_type: u32,
    place: vec3<f32>,
    color: vec3<f32>,
    angle: f32,
    casting_shadows: u32,
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
    @location(2) l_pos: vec4<f32>,
    @location(3) albedo: vec3<f32>,
};

@vertex
fn vs_main(v: VertexIn, i: Instance) -> VertexOutput {
    var model = mat4x4<f32>(i.model_c0, i.model_c1, i.model_c2, i.model_c3);
    var inv_model_t = mat4x4<f32>(i.model_invt_c0, i.model_invt_c1, i.model_invt_c2, i.model_invt_c3);

    var world_v = model * vec4<f32>(v.model_v, 1.0);
    var camera_v = projection * camera * world_v;

    var out: VertexOutput;
    out.position = camera_v;
    out.normal = normalize(inv_model_t * vec4(v.normal_v, 0.0));
    out.w_pos = world_v;
    out.albedo = i.albedo;

    return out;
}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var lightColor = vec3<f32>(1.0, 1.0, 1.0);

    var color = vec3(0.0, 0.0, 0.0);
    for (var i = 0; u32(i) < lights.length; i = i + 1) {
        var lightPos = lights.lights[i].place;
        var lightColor = lights.lights[i].color;

        var diffuseStrength = 0.5;
        var lightDir = normalize(lightPos - in.w_pos.xyz);

        var ambientStrength = 0.1;
        var ambient = ambientStrength * lightColor;

        var diffuseCoeff = max(dot(in.normal.xyz, lightDir), 0.0);
        var diffuse = diffuseStrength * diffuseCoeff * lightColor;

        var specularStrength = 0.4;
        var viewPos = camera_model[3].xyz;
        var viewDir = normalize(viewPos - in.w_pos.xyz);
        var reflectDir = reflect(-lightDir, in.normal.xyz);
        var specularCoeff = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        var specular = specularStrength * specularCoeff * lightColor;

        color += ambient + diffuse + specular;
    }

    return vec4(color, 1.0) * vec4(in.albedo, 1.0);
}
