#define_import_path gpubasics::deferred::ssao::bindings

@group(1) @binding(0) var<uniform> samples: array<vec3<f32>, 64>;
@group(1) @binding(1) var g_sampler: sampler;
@group(1) @binding(2) var noise_sampler: sampler;
@group(1) @binding(3) var g_normal: texture_2d<f32>;
@group(1) @binding(4) var t_noise: texture_2d<f32>;
@group(1) @binding(5) var g_depth: texture_depth_2d;

