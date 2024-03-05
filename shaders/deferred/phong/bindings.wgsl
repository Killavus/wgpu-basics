#define_import_path gpubasics::deferred::phong::bindings
#import gpubasics::phong::definitions::Lights;

@group(1) @binding(0) var<storage, read> lights: Lights;
@group(1) @binding(1) var g_sampler: sampler;
@group(1) @binding(2) var g_normal: texture_2d<f32>;
@group(1) @binding(3) var g_diffuse: texture_2d<f32>;
@group(1) @binding(4) var g_specular: texture_2d<f32>;
@group(1) @binding(5) var g_depth: texture_depth_2d;
