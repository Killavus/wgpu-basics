use anyhow::Result;
use encase::{ShaderSize, ShaderType, UniformBuffer};
use nalgebra as na;

use crate::gpu::Gpu;

type FVec4 = na::Vector4<f32>;

#[derive(Eq, PartialEq, Ord, PartialOrd, Clone, Copy, Debug)]
pub struct MaterialId(usize);

pub enum Material {
    PhongSolid {
        // w unused
        albedo: FVec4,
        // w unused
        diffuse: FVec4,
        // w = shininess
        specular: FVec4,
    },
}

#[derive(ShaderType)]
struct GpuPhongSolidRepr {
    albedo: FVec4,
    diffuse: FVec4,
    specular: FVec4,
}

enum GpuMaterial {
    PhongSolid {
        buffer: wgpu::Buffer,
        bind_group: wgpu::BindGroup,
    },
}

impl GpuMaterial {
    pub fn new(gpu: &Gpu, material: &Material, layouts: &MaterialAtlasLayouts) -> Result<Self> {
        use wgpu::util::DeviceExt;

        match material {
            Material::PhongSolid {
                albedo,
                diffuse,
                specular,
            } => {
                let repr_size: u64 = GpuPhongSolidRepr::SHADER_SIZE.into();
                let mut contents = UniformBuffer::new(Vec::with_capacity(repr_size as usize));
                contents.write(&GpuPhongSolidRepr {
                    albedo: *albedo,
                    diffuse: *diffuse,
                    specular: *specular,
                })?;

                let buffer = gpu
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Material::PhongSolid"),
                        contents: contents.into_inner().as_slice(),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });

                let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Material::PhongSolidBindGroup"),
                    layout: &layouts.phong_solid,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                });

                Ok(Self::PhongSolid {
                    buffer,
                    bind_group: bg,
                })
            }
        }
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        match self {
            Self::PhongSolid { bind_group, .. } => bind_group,
        }
    }
}

struct GpuPhongSolid {
    buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

pub struct MaterialAtlas {
    materials: Vec<Material>,
    gpu_materials: Vec<GpuMaterial>,
    pub layouts: MaterialAtlasLayouts,
}

pub struct MaterialAtlasLayouts {
    pub phong_solid: wgpu::BindGroupLayout,
}

impl MaterialAtlasLayouts {
    fn new(gpu: &Gpu) -> Self {
        let phong_solid = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("MaterialAtlas::PhongSolidLayout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        Self { phong_solid }
    }
}

impl MaterialAtlas {
    pub fn new(gpu: &Gpu) -> Self {
        Self {
            layouts: MaterialAtlasLayouts::new(gpu),
            materials: Vec::new(),
            gpu_materials: Vec::new(),
        }
    }

    pub fn add_material(&mut self, gpu: &Gpu, material: Material) -> Result<MaterialId> {
        let material_idx = self.materials.len();
        self.materials.push(material);
        self.gpu_materials.push(GpuMaterial::new(
            gpu,
            &self.materials[material_idx],
            &self.layouts,
        )?);

        Ok(MaterialId(material_idx))
    }

    pub fn bind_group(&self, material_id: MaterialId) -> &wgpu::BindGroup {
        self.gpu_materials[material_id.0].bind_group()
    }

    // pub fn update_material<F>(&mut self, material_id: MaterialId, updater: F)
    // where
    //     F: Fn(&mut Material),
    // {
    //     let material = &mut self.materials[material_id.0];
    //     updater(material);
    // }
}
