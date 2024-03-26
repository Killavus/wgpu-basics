use winit::window::Window;

use crate::{
    gpu::Gpu, light_scene::LightScene, material::MaterialAtlas, scene::GpuScene,
    scene_uniform::SceneUniform, shader_compiler::ShaderCompiler,
};

pub struct RenderContext<'window> {
    pub gpu: Gpu<'window>,
    pub shader_compiler: ShaderCompiler,
    pub gpu_scene: GpuScene,
    pub light_scene: LightScene,
    pub scene_uniform: SceneUniform,
    pub material_atlas: MaterialAtlas,
    pub window: &'window Window,
}

impl<'window> RenderContext<'window> {
    pub fn new(
        window: &'window Window,
        gpu: Gpu<'window>,
        shader_compiler: ShaderCompiler,
        scene_uniform: SceneUniform,
        gpu_scene: GpuScene,
        material_atlas: MaterialAtlas,
        light_scene: LightScene,
    ) -> Self {
        Self {
            window,
            gpu,
            shader_compiler,
            scene_uniform,
            gpu_scene,
            material_atlas,
            light_scene,
        }
    }
}
