@group(0) @binding(0) var<uniform> camera: mat4x4<f32>;
@group(0) @binding(1) var<uniform> projection: mat4x4<f32>;
@group(1) @binding(0) var skybox_texture: texture_cube<f32>;
@group(1) @binding(1) var skybox_sampler: sampler;

struct VertexIn {
    @location(0) model_v: vec3<f32>,
    @location(1) normal_v: vec3<f32>,
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coord: vec3<f32>,
};


@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var o: VertexOut;

    var camera_mat = mat4x4<f32>(
        camera[0],
        camera[1],
        camera[2],
        vec4<f32>(0.0, 0.0, 0.0, 1.0)
    );

    var cam_v = projection * camera_mat * vec4<f32>(v.model_v, 1.0);
    o.position = cam_v.xyww;
    o.tex_coord = v.model_v;

    return o;
}

@fragment
fn fs_main(i: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(skybox_texture, skybox_sampler, i.tex_coord);
}
