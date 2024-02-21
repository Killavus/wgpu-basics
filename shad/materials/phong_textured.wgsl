#define_import_path gpubasics::materials::phong_textured

@group(2) @binding(0) var diffuse_t: texture_2d<f32>;
@group(2) @binding(1) var specular_t: texture_2d<f32>;
#ifdef NORMAL_MAP
@group(2) @binding(2) var normal_t: texture_2d<f32>;
@group(2) @binding(3) var mat_sampler: sampler;
@group(2) @binding(4) var<uniform> shininess: f32;
#else
@group(2) @binding(2) var mat_sampler: sampler;
@group(2) @binding(3) var<uniform> shininess: f32;
#endif
