use anyhow::Result;
use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderDefValue,
};

pub struct ShaderCompiler {
    composer: Composer,
}

impl ShaderCompiler {
    pub fn new() -> Result<Self> {
        let mut composer = Composer::default();

        let files = &[
            "./shad/vertex_data.wgsl",
            "./shad/global_bindings.wgsl",
            "./shad/instances/model.wgsl",
            "./shad/phong/vertex_output.wgsl",
            "./shad/phong/light_defs.wgsl",
            "./shad/materials/phong_solid.wgsl",
            "./shad/materials/phong_textured.wgsl",
            "./shad/phong/material_bindings.wgsl",
            "./shad/phong/cascaded_shadow_map.wgsl",
            "./shad/phong/bindings.wgsl",
            "./shad/phong/light_functions.wgsl",
        ];

        for file in files {
            let content = std::fs::read_to_string(file).unwrap();
            composer.add_composable_module(ComposableModuleDescriptor {
                source: &content,
                file_path: file,
                language: naga_oil::compose::ShaderLanguage::Wgsl,
                ..Default::default()
            })?;
        }

        Ok(Self { composer })
    }

    pub fn compile(
        &mut self,
        path: &str,
        shader_defs: Vec<(String, ShaderDefValue)>,
    ) -> Result<wgpu::naga::Module> {
        use std::collections::HashMap;
        use std::fs;

        let module = self
            .composer
            .make_naga_module(NagaModuleDescriptor {
                source: &fs::read_to_string(path)?,
                file_path: path,
                shader_type: naga_oil::compose::ShaderType::Wgsl,
                shader_defs: HashMap::from_iter(shader_defs),
                additional_imports: &[],
            })
            .inspect_err(|e| eprintln!("{}", e.emit_to_string(&self.composer)))?;

        Ok(module)
    }
}
