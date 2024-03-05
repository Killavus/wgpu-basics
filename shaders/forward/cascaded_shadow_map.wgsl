#import gpubasics::forward::buffers::vertex::{Vertex};
#import gpubasics::forward::buffers::instance::{Instance, model};

#import gpubasics::global::bindings::{camera, projection};

@vertex
fn vs_main(v: Vertex, i: Instance) -> @builtin(position) vec4<f32> {
    var model = model(i);

    var world_v = model * vec4<f32>(v.model_v, 1.0);
    var camera_v = projection * camera * world_v;

    return camera_v;
}
