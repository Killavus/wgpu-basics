@group(0) @binding(0) var output: texture_storage_2d<r8unorm, write>;
@group(0) @binding(1) var input: texture_2d<f32>;
@group(0) @binding(2) var tex_sampler: sampler;
struct Flip {
    value: u32
};
struct FilterSize {
    value: u32
};

@group(0) @binding(3) var<uniform> flip: Flip;
@group(0) @binding(4) var<uniform> filter_size: FilterSize;

// Every thread is fetching 4x4 piece of a texture.
// There are 32 threads in a workgroup, so we are having 128x4 pixels fetched.
var<workgroup> shared_mem: array<array<vec3f, 128>, 4>;

@compute @workgroup_size(32, 1, 1)
fn blur(@builtin(workgroup_id) WorkGroupID: vec3u, @builtin(local_invocation_id) LocalInvocationID: vec3u) {
    var imageDim = textureDimensions(input);
    var filterCenter = (filter_size.value - 1) / 2;
    var baseIndex = WorkGroupID.xy * vec2(128 - filter_size.value - 1, 4) + LocalInvocationID.xy * vec2(4, 1) - vec2(filterCenter, 0);

    for (var r = 0; r < 4; r += 1) {
        for (var c = 0; c < 4; c += 1) {
            var coord = baseIndex + vec2(u32(c), u32(r));
            if flip.value == 1u {
                coord = coord.xy;
            }

            shared_mem[r][4 * LocalInvocationID.x + u32(c)] = textureSampleLevel(input, tex_sampler, (vec2f(coord) + vec2f(0.25, 0.25)) / vec2f(imageDim), 0.0).rgb;
        }
    }

    workgroupBarrier();

    for (var r = 0; r < 4; r += 1) {
        for (var c = 0; c < 4; c += 1) {
            var writeIndex = baseIndex + vec2(u32(c), u32(r));
            if flip.value == 1u {
                writeIndex = writeIndex.xy;
            }

            let center = i32(4 * LocalInvocationID.x + u32(c));
            if center >= i32(filterCenter) && center < 128 - i32(filterCenter) && all(writeIndex < imageDim) {
                var acc = vec3(0.0, 0.0, 0.0);
                for (var i = 0; u32(i) < filter_size.value; i += 1) {
                    var f = center + i - i32(filterCenter);
                    acc += (1.0 / f32(filter_size.value)) * shared_mem[r][f];
                }
                textureStore(output, writeIndex, vec4(acc, 1.0));
            }
        }
    }
}
