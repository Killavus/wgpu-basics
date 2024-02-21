#define_import_path gpubasics::phong::light_functions
#import gpubasics::phong::bindings::lights;
#import gpubasics::phong::light_defs::{Light};
#import gpubasics::phong::vertex_output::VertexOutput;

fn calculateDirectional(in: VertexOutput, light: Light) -> vec3<f32> {
    return vec3(1.0, 0.0, 0.0);
}

fn calculateSpot(in: VertexOutput, light: Light) -> vec3<f32> {
    return vec3(0.0, 1.0, 0.0);
}

fn calculatePoint(in: VertexOutput, light: Light) -> vec3<f32> {
    return vec3(0.0, 0.0, 1.0);
}

fn fragmentLight(in: VertexOutput) -> vec3<f32> {
    var color = vec3(0.0, 0.0, 0.0);

    for (var i = 0; u32(i) < lights.num_directional; i = i + 1) {
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
