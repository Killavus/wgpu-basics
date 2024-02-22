use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderDefValue,
};

// TODO: Make its interface public.
fn compose_shad() -> wgpu::naga::Module {
    let mut composer = Composer::default();

    let files = &[
        "./shad/instances/model.wgsl",
        "./shad/materials/phong_solid.wgsl",
        "./shad/phong/vertex_output.wgsl",
        "./shad/phong/light_defs.wgsl",
        "./shad/phong/bindings.wgsl",
        "./shad/phong/light_functions.wgsl",
        "./shad/materials/phong_textured.wgsl",
        "./shad/global_bindings.wgsl",
        "./shad/vertex_data.wgsl",
    ];

    for file in files {
        let content = std::fs::read_to_string(file).unwrap();
        composer
            .add_composable_module(ComposableModuleDescriptor {
                source: &content,
                file_path: file,
                language: naga_oil::compose::ShaderLanguage::Wgsl,
                ..Default::default()
            })
            .unwrap();
    }

    let module = composer.make_naga_module(NagaModuleDescriptor {
        source: &std::fs::read_to_string("./shad/test.wgsl").unwrap(),
        file_path: "./shad/test.wgsl",
        shader_type: naga_oil::compose::ShaderType::Wgsl,
        shader_defs: [
            ("VERTEX_PN".into(), ShaderDefValue::Bool(true)),
            ("MATERIAL_PHONG_SOLID".into(), ShaderDefValue::Bool(true)),
        ]
        .into(),
        ..Default::default()
    });

    if let Err(e) = module.as_ref() {
        println!("{}", e.emit_to_string(&composer));
    }

    module.unwrap()
}
