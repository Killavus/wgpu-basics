#define_import_path gpubasics::phong::light_functions
#import gpubasics::global::bindings::camera_model;
#import gpubasics::phong::bindings::lights;
#import gpubasics::phong::light_defs::{Light};

#ifdef DEFERRED
#import gpubasics::deferred::vertex_output::VertexOutput;
#import gpubasics::deferred::functions::{normal, diffuse as materialDiffuse, specular as materialSpecular, ambient as materialAmbient, worldPos};

fn shininess(in: VertexOutput) -> f32 {
    return 32.0;
}
#else
#import gpubasics::phong::vertex_output::VertexOutput;
#import gpubasics::phong::material_bindings::{normal, materialDiffuse, materialSpecular, materialAmbient, shininess};

fn worldPos(in: VertexOutput) -> vec4<f32> {
    return in.w_pos;
}
#endif

#ifdef SHADOW_MAP
#import gpubasics::phong::cascaded_shadow_map::calculateShadow;
#endif

fn attenuation(lightDistance: f32, light: Light) -> f32 {
    var attenuationConstant = light.ambient.w;
    var attenuationLinear = light.diffuse.w;
    var attenuationQuadratic = light.specular.w;

    return 1.0 / (attenuationConstant + attenuationLinear * lightDistance + attenuationQuadratic * lightDistance * lightDistance);
}

fn phongLighting(in: VertexOutput, lightDirection: vec3<f32>, attenuation: f32, light: Light, notShadowed: f32) -> vec3<f32> {
    var color = vec3(0.0, 0.0, 0.0);
    var lAmbient = light.ambient.xyz;
    var lDiffuse = light.diffuse.xyz;
    var lSpecular = light.specular.xyz;

    var n = normal(in);
    var mAmbient = materialAmbient(in);
    var mDiffuse = materialDiffuse(in);
    var mSpecular = materialSpecular(in);
    var mShininess = shininess(in);

    var viewPosition = camera_model[3].xyz;
    var viewDirection = normalize(viewPosition - worldPos(in).xyz);
    var halfway = normalize(lightDirection + viewDirection);

    color += lAmbient * mAmbient;
    var diffuseCoeff = max(dot(n, lightDirection), 0.0);
    color += notShadowed * mDiffuse * attenuation * diffuseCoeff * lDiffuse;

    var specularCoeff = max(pow(max(dot(n, halfway), 0.0), mShininess), 0.0);
    color += notShadowed * mSpecular * attenuation * specularCoeff * lSpecular;

    return color;
}

fn calculateDirectional(in: VertexOutput, light: Light) -> vec3<f32> {
    var lightDirection = -light.direction.xyz;
    var attenuation = 1.0;

    #ifdef SHADOW_MAP
    var notShadowed = 1.0 - calculateShadow(in, lightDirection);
    #else
    var notShadowed = 1.0;
    #endif

    return phongLighting(in, lightDirection, attenuation, light, notShadowed);
}

fn calculateSpot(in: VertexOutput, light: Light) -> vec3<f32> {
    var fragmentToLight = light.position.xyz - worldPos(in).xyz;
    var lightDirection = normalize(fragmentToLight);
    var lightDistance = length(fragmentToLight);

    var attenuation = attenuation(lightDistance, light);

    var spotDirection = -light.direction.xyz;
    var theta = dot(lightDirection, spotDirection);
    var angle = light.position.w;
    var epsilon = cos(angle);

    if theta <= epsilon {
        return vec3(0.0, 0.0, 0.0);
    } else {
        return phongLighting(in, lightDirection, attenuation, light, 1.0);
    }
}

fn calculatePoint(in: VertexOutput, light: Light) -> vec3<f32> {
    var fragmentToLight = light.position.xyz - worldPos(in).xyz;
    var lightDirection = normalize(fragmentToLight);
    var lightDistance = length(fragmentToLight);

    var attenuation = attenuation(lightDistance, light);

    return phongLighting(in, lightDirection, attenuation, light, 1.0);
}

fn fragmentLight(in: VertexOutput) -> vec3<f32> {
    var color = vec3(0.0, 0.0, 0.0);

    for (var i = u32(0); i < lights.num_directional; i = i + 1) {
        color += calculateDirectional(in, lights.lights[i]);
    }

    for (var i = u32(0); i < lights.num_point; i = i + 1) {
        color += calculatePoint(in, lights.lights[i + lights.num_directional]);
    }

    for (var i = u32(0); i < lights.num_spot; i = i + 1) {
        color += calculateSpot(in, lights.lights[i + lights.num_directional + lights.num_point]);
    }

    return color;
}
