#define_import_path gpubasics::phong::material_bindings

#ifdef MATERIAL_PHONG_SOLID
#import gpubasics::materials::phong_solid::{normal as fNorm, materialDiffuse as fDiff, materialSpecular as fSpec, materialAmbient as fAmb, shininess as fShin};
#endif

#ifdef MATERIAL_PHONG_TEXTURED
#import gpubasics::materials::phong_textured::{normal as fNorm, materialDiffuse as fDiff, materialSpecular as fSpec, materialAmbient as fAmb, shininess as fShin};
#endif

#import gpubasics::phong::vertex_output::VertexOutput;

fn normal(in: VertexOutput) -> vec3<f32> {
    return fNorm(in);
}

fn materialDiffuse(in: VertexOutput) -> vec3<f32> {
    return fDiff(in);
}

fn materialSpecular(in: VertexOutput) -> vec3<f32> {
    return fSpec(in);
}

fn materialAmbient(in: VertexOutput) -> vec3<f32> {
    return fAmb(in);
}

fn shininess(in: VertexOutput) -> f32 {
    return fShin(in);
}
