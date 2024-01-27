@group(0) @binding(0)
var texture: texture_2d<f32>;
@group(0) @binding(1)
var textureSampler: sampler;

struct PostProcessSettings {
    b_c_s_g: vec4<f32>
}

@group(0) @binding(2) var<uniform> settings: PostProcessSettings;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    var VERTEX: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0)
    );

    var TEX: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0)
    );

    out.clip_position = vec4<f32>(VERTEX[in_vertex_index], 0.0, 1.0);
    out.tex_coords = vec2<f32>(TEX[in_vertex_index]);

    return out;
}

fn grayscaleAvg(color: vec3<f32>) -> vec3<f32> {
    let avg = color.x + color.y + color.z;

    return vec3<f32>(avg, avg, avg) / 3.0;
}

fn contrastBrightness(c: f32, b: f32, color: vec3<f32>) -> vec3<f32> {
    return saturate((color - 0.5) * c + 0.5 + b);
}

fn gamma(color: vec3<f32>, gamma: f32) -> vec3<f32> {
    return pow(color, vec3<f32>(gamma));
}

fn saturation(color: vec3<f32>, s: f32) -> vec3<f32> {
    // This is perceptual grayscale, which accounts for the greener color more,
    // since it contributes to the brightness of the grayscale the most.
    var grayscale = dot(color, vec3<f32>(0.299, 0.587, 0.114));

    return saturate(mix(vec3(grayscale, grayscale, grayscale), color, s));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = textureSample(texture, textureSampler, in.tex_coords);
    var brightness = settings.b_c_s_g.x;
    var contrast = settings.b_c_s_g.y;
    var saturation = settings.b_c_s_g.z;
    var gamma = settings.b_c_s_g.w;

    return vec4<f32>(gamma(saturation(contrastBrightness(brightness, contrast, color.xyz), saturation), gamma), 1.0);
}
