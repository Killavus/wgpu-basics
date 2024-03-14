#import gpubasics::deferred::shaders::screen_quad_vs::screenQuad;
#import gpubasics::deferred::outputs::vertex::{VertexOutput};
#import gpubasics::deferred::ssao::fragment::{worldPos, normal, noise, depth};
#import gpubasics::deferred::ssao::bindings::samples;
#import gpubasics::global::bindings::{camera, projection};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    return screenQuad(in_vertex_index);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) f32 {
    var SSAO_SAMPLES_CNT = u32(64);

    var pos = worldPos(in).xyz;
    var normal = normal(in);
    var noise = noise(in).rgb;

    // Gram-Schmidt orthogonalization
    var tangent = normalize(noise - normal * dot(noise, normal));
    var bitangent = cross(normal, tangent);

    var tbn = mat3x3(tangent, bitangent, normal);
    var radius = 0.33;

    var occlusion = 0.0;
    for (var i = u32(0); i < SSAO_SAMPLES_CNT; i += u32(1)) {
        var sample = tbn * samples[i];
        sample = pos + sample * radius;

        var offset = vec4(sample, 1.0);
        var clipPos = projection * camera * offset;
        clipPos /= clipPos.w;

        var sampleOut: VertexOutput;
        sampleOut.clip = vec4(clipPos.xy, 0.0, 1.0);
        sampleOut.uv = clipPos.xy * vec2(0.5, -0.5) + 0.5;

        var sampleDepth = worldPos(sampleOut).z;
        var rangeCheck = smoothstep(0.0, 1.0, radius / abs(pos.z - sampleDepth));

        if sampleDepth >= sample.z + 0.025 {
            occlusion += 1.0 * rangeCheck;
        }
    }

    occlusion = 1.0 - (occlusion / f32(SSAO_SAMPLES_CNT));
    return occlusion;
}
