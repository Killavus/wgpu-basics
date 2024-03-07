#define_import_path gpubasics::phong::fragment

#ifdef DEFERRED
#import gpubasics::deferred::outputs::vertex::VertexOutput;
#import gpubasics::deferred::phong::fragment::{normal, worldPos, cameraPos, diffuse as materialDiffuse, diffuse as materialAmbient, specular as materialSpecular, shininess, ambientOcclusion};
#else
#import gpubasics::forward::outputs::vertex::{worldPos, cameraPos, VertexOutput};
#ifdef MATERIAL_PHONG_SOLID
#import gpubasics::materials::phong_solid::{normal, materialDiffuse, materialSpecular, materialAmbient, shininess};
#endif

#ifdef MATERIAL_PHONG_TEXTURED
#import gpubasics::materials::phong_textured::{normal, materialDiffuse, materialSpecular, materialAmbient, shininess};
#endif
#endif

fn fragmentWorldPos(in: VertexOutput) -> vec4<f32> {
    return worldPos(in);
}

fn fragmentCameraPos(in: VertexOutput) -> vec4<f32> {
    return cameraPos(in);
}

fn fragmentNormal(in: VertexOutput) -> vec3<f32> {
    return normal(in);
}

fn fragmentDiffuse(in: VertexOutput) -> vec3<f32> {
    return materialDiffuse(in);
}

fn fragmentSpecular(in: VertexOutput) -> vec3<f32> {
    return materialSpecular(in);
}

fn fragmentAmbient(in: VertexOutput) -> vec3<f32> {
    return materialAmbient(in);
}

fn fragmentShininess(in: VertexOutput) -> f32 {
    return shininess(in);
}

fn fragmentOcclusion(in: VertexOutput) -> f32 {
    #ifdef DEFERRED
    return ambientOcclusion(in);
    #else
    return 1.0;
    #endif
}
