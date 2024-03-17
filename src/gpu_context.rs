use anyhow::Result;
use fnv::FnvHashMap;
use std::sync::RwLock;

use crate::camera::GpuCamera;
use crate::gpu::Gpu;
use crate::material::MaterialAtlas;
use crate::projection::GpuProjection;
use crate::scene::GpuScene;
use crate::scene_uniform::SceneUniform;
use crate::shader_compiler::ShaderCompiler;

pub struct RenderContext<'window> {
    gpu: Gpu<'window>,
    shader_compiler: ShaderCompiler,
    scene_uniform: SceneUniform,
    material_atlas: RwLock<MaterialAtlas>,
    scene: RwLock<GpuScene>,
}

impl<'window> RenderContext<'window> {
    fn init(
        gpu: Gpu<'window>,
        shader_compiler: ShaderCompiler,
        scene_uniform: SceneUniform,
        material_atlas: MaterialAtlas,
        scene: GpuScene,
    ) -> Self {
        Self {
            gpu,
            shader_compiler,
            scene_uniform,
            material_atlas: RwLock::new(material_atlas),
            scene: RwLock::new(scene),
        }
    }

    pub async fn create(
        window: &'window winit::window::Window,
        camera: GpuCamera,
        projection: GpuProjection,
        scene: GpuScene,
    ) -> Result<Self> {
        let gpu = Gpu::from_window(window).await?;
        let shader_compiler = ShaderCompiler::new("./shaders")?;
        let scene_uniform = SceneUniform::new(&gpu, &camera, &projection);
        let material_atlas = MaterialAtlas::new(&gpu);

        Ok(Self::init(
            gpu,
            shader_compiler,
            scene_uniform,
            material_atlas,
            scene,
        ))
    }
}
