use std::path::Path;

use anyhow::Result;
use encase::{ShaderSize, ShaderType, UniformBuffer};
use nalgebra as na;

use crate::gpu::Gpu;

type FVec4 = na::Vector4<f32>;

#[derive(Eq, PartialEq, Ord, PartialOrd, Clone, Copy, Debug)]
pub struct MaterialId(usize);

#[allow(clippy::enum_variant_names)]
pub enum Material {
    PhongSolid {
        // w unused
        ambient: FVec4,
        // w unused
        diffuse: FVec4,
        // w = shininess
        specular: FVec4,
    },
    PhongTextured {
        diffuse: wgpu::Texture,
        specular: Option<wgpu::Texture>,
    },
    PhongTexturedNormal {
        diffuse: wgpu::Texture,
        normal: wgpu::Texture,
        specular: Option<wgpu::Texture>,
    },
}

#[derive(ShaderType)]
struct GpuPhongSolidRepr {
    ambient: FVec4,
    diffuse: FVec4,
    specular: FVec4,
}

#[allow(clippy::enum_variant_names)]
enum GpuMaterial {
    PhongSolid {
        buffer: wgpu::Buffer,
        bind_group: wgpu::BindGroup,
    },
    PhongTextured {
        bind_group: wgpu::BindGroup,
    },
    PhongTexturedNormal {
        bind_group: wgpu::BindGroup,
    },
}

impl GpuMaterial {
    pub fn new(
        gpu: &Gpu,
        material: &Material,
        default_textures: &MaterialAtlasTextureDefaults,
        layouts: &MaterialAtlasLayouts,
    ) -> Result<Self> {
        use wgpu::util::DeviceExt;

        match material {
            Material::PhongSolid {
                ambient,
                diffuse,
                specular,
            } => {
                let repr_size: u64 = GpuPhongSolidRepr::SHADER_SIZE.into();
                let mut contents = UniformBuffer::new(Vec::with_capacity(repr_size as usize));
                contents.write(&GpuPhongSolidRepr {
                    ambient: *ambient,
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
            Material::PhongTextured { diffuse, specular } => {
                let diffuse_view = diffuse.create_view(&wgpu::TextureViewDescriptor::default());
                let specular_view = specular
                    .as_ref()
                    .unwrap_or(&default_textures.black)
                    .create_view(&wgpu::TextureViewDescriptor::default());

                let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Material::PhongTexturedBindGroup"),
                    layout: &layouts.phong_textured,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&diffuse_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&specular_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&default_textures.sampler),
                        },
                    ],
                });

                Ok(Self::PhongTextured { bind_group: bg })
            }
            Material::PhongTexturedNormal {
                diffuse,
                specular,
                normal,
            } => {
                let diffuse_view = diffuse.create_view(&wgpu::TextureViewDescriptor::default());
                let normal_view = normal.create_view(&wgpu::TextureViewDescriptor::default());
                let specular_view = specular
                    .as_ref()
                    .unwrap_or(&default_textures.black)
                    .create_view(&wgpu::TextureViewDescriptor::default());

                let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Material::PhongTexturedNormalBindGroup"),
                    layout: &layouts.phong_textured_normal,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&diffuse_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&specular_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&normal_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(&default_textures.sampler),
                        },
                    ],
                });

                Ok(Self::PhongTextured { bind_group: bg })
            }
        }
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        match self {
            Self::PhongSolid { bind_group, .. } => bind_group,
            Self::PhongTextured { bind_group, .. } => bind_group,
            Self::PhongTexturedNormal { bind_group, .. } => bind_group,
        }
    }
}

pub struct MaterialAtlas {
    materials: Vec<Material>,
    gpu_materials: Vec<GpuMaterial>,
    pub textures: MaterialAtlasTextureDefaults,
    pub layouts: MaterialAtlasLayouts,
}

pub struct MaterialAtlasLayouts {
    pub phong_solid: wgpu::BindGroupLayout,
    pub phong_textured: wgpu::BindGroupLayout,
    pub phong_textured_normal: wgpu::BindGroupLayout,
}

pub struct MaterialAtlasTextureDefaults {
    pub white: wgpu::Texture,
    pub black: wgpu::Texture,
    sampler: wgpu::Sampler,
}

impl MaterialAtlasTextureDefaults {
    pub fn new(gpu: &Gpu) -> Self {
        let white = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MaterialAtlas::WhiteTexture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let black = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MaterialAtlas::BlackTexture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("MaterialAtlas::TextureSampler"),
            address_mode_u: wgpu::AddressMode::MirrorRepeat,
            address_mode_v: wgpu::AddressMode::MirrorRepeat,
            address_mode_w: wgpu::AddressMode::MirrorRepeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        gpu.queue.write_texture(
            black.as_image_copy(),
            &[0, 0, 0, 255],
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        gpu.queue.write_texture(
            white.as_image_copy(),
            &[255, 255, 255, 255],
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        Self {
            white,
            black,
            sampler,
        }
    }
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

        let phong_textured =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("MaterialAtlas::PhongTexturedLayout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

        let phong_textured_normal =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("MaterialAtlas::PhongTexturedNormalLayout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

        Self {
            phong_solid,
            phong_textured,
            phong_textured_normal,
        }
    }
}

impl MaterialAtlas {
    pub fn new(gpu: &Gpu) -> Self {
        Self {
            layouts: MaterialAtlasLayouts::new(gpu),
            textures: MaterialAtlasTextureDefaults::new(gpu),
            materials: Vec::new(),
            gpu_materials: Vec::new(),
        }
    }

    pub fn add_phong_solid(
        &mut self,
        gpu: &Gpu,
        ambient: FVec4,
        diffuse: FVec4,
        specular: FVec4,
    ) -> Result<MaterialId> {
        let material = Material::PhongSolid {
            ambient,
            diffuse,
            specular,
        };

        self.add_material(gpu, material)
    }

    pub fn add_phong_textured(
        &mut self,
        gpu: &Gpu,
        diffuse: impl AsRef<Path>,
        specular: Option<impl AsRef<Path>>,
    ) -> Result<MaterialId> {
        let diffuse = Self::gpu_texture(gpu, Self::load_texture(diffuse)?);
        let specular = match specular {
            Some(path) => Some(Self::gpu_texture(gpu, Self::load_texture(path)?)),
            None => None,
        };

        self.add_material(gpu, Material::PhongTextured { diffuse, specular })
    }

    pub fn add_phong_textured_normal(
        &mut self,
        gpu: &Gpu,
        diffuse: impl AsRef<Path>,
        specular: Option<impl AsRef<Path>>,
        normal: impl AsRef<Path>,
    ) -> Result<MaterialId> {
        let diffuse = Self::gpu_texture(gpu, Self::load_texture(diffuse)?);
        let normal = Self::gpu_texture(gpu, Self::load_texture(normal)?);
        let specular = match specular {
            Some(path) => Some(Self::gpu_texture(gpu, Self::load_texture(path)?)),
            None => None,
        };

        self.add_material(
            gpu,
            Material::PhongTexturedNormal {
                diffuse,
                specular,
                normal,
            },
        )
    }

    pub fn is_normal_mapped(&self, material_id: MaterialId) -> bool {
        matches!(
            self.materials[material_id.0],
            Material::PhongTexturedNormal { .. }
        )
    }

    fn load_texture(path: impl AsRef<Path>) -> Result<image::RgbaImage> {
        let img = image::open(path)?;

        Ok(img.to_rgba8())
    }

    fn gpu_texture(gpu: &Gpu, image: image::RgbaImage) -> wgpu::Texture {
        use image::EncodableLayout;
        let (width, height) = image.dimensions();

        let tex_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: tex_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        gpu.queue.write_texture(
            texture.as_image_copy(),
            image.as_bytes(),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            tex_size,
        );

        texture
    }

    fn add_material(&mut self, gpu: &Gpu, material: Material) -> Result<MaterialId> {
        let material_idx = self.materials.len();
        self.materials.push(material);
        self.gpu_materials.push(GpuMaterial::new(
            gpu,
            &self.materials[material_idx],
            &self.textures,
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
