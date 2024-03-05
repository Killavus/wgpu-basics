#define_import_path gpubasics::shadow::cascaded::bindings
#import gpubasics::shadow::cascaded::definitions::{ShadowMapMatrices, ShadowMapResult};

#ifdef DEFERRED
@group(2) @binding(0) var<uniform> smap_matrices: ShadowMapMatrices;
@group(2) @binding(1) var smap_sampler: sampler;
@group(2) @binding(2) var smap: texture_depth_2d_array;
@group(2) @binding(3) var<uniform> smap_result: ShadowMapResult;
#else
@group(3) @binding(0) var<uniform> smap_matrices: ShadowMapMatrices;
@group(3) @binding(1) var smap_sampler: sampler;
@group(3) @binding(2) var smap: texture_depth_2d_array;
@group(3) @binding(3) var<uniform> smap_result: ShadowMapResult;
#endif
