#define_import_path gpubasics::phong::bindings
#import gpubasics::phong::light_defs::Lights;

@group(1) @binding(0) var<storage, read> lights: Lights;

#ifdef MATERIAL_PHONG_SOLID
#import gpubasics::materials::phong_solid;
#endif

#ifdef MATERIAL_PHONG_TEXTURED
#import gpubasics::materials::phong_textured;
#endif
