
#define_import_path gpubasics::instances::model

#ifdef VERTEX_PN
struct Instance {
    @location(2) model_ca: vec4<f32>,
    @location(3) model_cb: vec4<f32>,
    @location(4) model_cc: vec4<f32>,
    @location(5) model_cd: vec4<f32>,
    @location(6) model_invt_ca: vec4<f32>,
    @location(7) model_invt_cb: vec4<f32>,
    @location(8) model_invt_cc: vec4<f32>,
    @location(9) model_invt_cd: vec4<f32>,
};
#endif

#ifdef VERTEX_PNUV
struct Instance {
    @location(3) model_ca: vec4<f32>,
    @location(4) model_cb: vec4<f32>,
    @location(5) model_cc: vec4<f32>,
    @location(6) model_cd: vec4<f32>,
    @location(7) model_invt_ca: vec4<f32>,
    @location(8) model_invt_cb: vec4<f32>,
    @location(9) model_invt_cc: vec4<f32>,
    @location(10) model_invt_cd: vec4<f32>,
};
#endif

#ifdef VERTEX_PNTBUV
struct Instance {
    @location(5) model_ca: vec4<f32>,
    @location(6) model_cb: vec4<f32>,
    @location(7) model_cc: vec4<f32>,
    @location(8) model_cd: vec4<f32>,
    @location(9) model_invt_ca: vec4<f32>,
    @location(10) model_invt_cb: vec4<f32>,
    @location(11) model_invt_cc: vec4<f32>,
    @location(12) model_invt_cd: vec4<f32>,
};
#endif

fn model(instance: Instance) -> mat4x4<f32> {
    return mat4x4<f32>(
        instance.model_ca,
        instance.model_cb,
        instance.model_cc,
        instance.model_cd,
    );
}

fn model_invt(instance: Instance) -> mat4x4<f32> {
    return mat4x4<f32>(
        instance.model_invt_ca,
        instance.model_invt_cb,
        instance.model_invt_cc,
        instance.model_invt_cd,
    );
}
