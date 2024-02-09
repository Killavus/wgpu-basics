use anyhow::{Context, Result};
use nalgebra as na;
use std::path::Path;

use crate::{
    gpu::Gpu,
    material::{MaterialAtlas, MaterialId, SpecularTexture},
    mesh::{Geometry, Mesh, MeshBuilder, NormalSource, TangentSpaceInformation},
};

pub struct ObjLoader;

fn flat_to_v3(v: &[f32]) -> Vec<na::Vector3<f32>> {
    v.chunks(3)
        .map(|c| na::Vector3::new(c[0], c[1], c[2]))
        .collect()
}

fn flat_to_v2(v: &[f32]) -> Vec<na::Vector2<f32>> {
    v.chunks(2).map(|c| na::Vector2::new(c[0], c[1])).collect()
}

pub struct ObjLoaderSettings {
    pub calculate_tangent_space: bool,
}

impl ObjLoader {
    pub fn load(
        path: impl AsRef<Path>,
        gpu: &Gpu,
        material_atlas: &mut MaterialAtlas,
        settings: ObjLoaderSettings,
    ) -> Result<(Vec<Mesh>, Vec<MaterialId>)> {
        let (models, materials) = tobj::load_obj(path.as_ref(), &tobj::LoadOptions::default())
            .context("failed to load obj file")?;

        let materials = materials?;

        let mut local_materials = Vec::with_capacity(materials.len());

        for material in materials.iter() {
            let is_phong_solid = material.diffuse.is_some() && material.ambient.is_some();
            let is_phong_textured = material.diffuse_texture.is_some();
            let is_phong_textured_normal = is_phong_textured && material.normal_texture.is_some();

            if is_phong_solid {
                let ambient = material.ambient.unwrap();
                let ambient = na::Vector4::new(ambient[0], ambient[1], ambient[2], 0.0);
                let diffuse_f = material.diffuse.unwrap();
                let diffuse = na::Vector4::new(diffuse_f[0], diffuse_f[1], diffuse_f[2], 0.0);
                let specular = material.specular.unwrap_or(diffuse_f);
                let specular = na::Vector4::new(specular[0], specular[1], specular[2], 0.0);

                local_materials.push((
                    material.name.clone(),
                    material_atlas.add_phong_solid(gpu, ambient, diffuse, specular)?,
                ));
            } else if is_phong_textured_normal {
                let diffuse_texture = material
                    .diffuse_texture
                    .as_ref()
                    .map(|tex_path| {
                        let base_path = path.as_ref().parent().unwrap_or(path.as_ref());
                        base_path.join(tex_path)
                    })
                    .unwrap();

                let specular = material
                    .specular_texture
                    .as_ref()
                    .map(|tex_path| {
                        let base_path = path.as_ref().parent().unwrap_or(path.as_ref());
                        SpecularTexture::Provided(
                            base_path.join(tex_path).to_str().unwrap().to_owned(),
                            32.0,
                        )
                    })
                    .unwrap_or(SpecularTexture::FullDiffuse);

                let normal = material
                    .normal_texture
                    .as_ref()
                    .map(|tex_path| {
                        let base_path = path.as_ref().parent().unwrap_or(path.as_ref());
                        base_path.join(tex_path)
                    })
                    .unwrap();

                local_materials.push((
                    material.name.clone(),
                    material_atlas.add_phong_textured_normal(
                        gpu,
                        &diffuse_texture,
                        specular,
                        &normal,
                    )?,
                ));
            } else if is_phong_textured {
                let diffuse_texture = material
                    .diffuse_texture
                    .as_ref()
                    .map(|tex_path| {
                        let base_path = path.as_ref().parent().unwrap_or(path.as_ref());
                        base_path.join(tex_path)
                    })
                    .unwrap();

                let specular = material
                    .specular_texture
                    .as_ref()
                    .map(|tex_path| {
                        let base_path = path.as_ref().parent().unwrap_or(path.as_ref());
                        SpecularTexture::Provided(
                            base_path.join(tex_path).to_str().unwrap().to_owned(),
                            32.0,
                        )
                    })
                    .unwrap_or(SpecularTexture::FullDiffuse);

                local_materials.push((
                    material.name.clone(),
                    material_atlas.add_phong_textured(gpu, &diffuse_texture, specular)?,
                ));
            }
        }

        let mut mesh_materials = vec![];
        let mut meshes = vec![];

        for (idx, model) in models.into_iter().enumerate() {
            let mut tan_space_info = None;
            if settings.calculate_tangent_space
                && material_atlas.is_normal_mapped(local_materials[idx].1)
            {
                tan_space_info = Some(TangentSpaceInformation {
                    texture_uvs: flat_to_v2(&model.mesh.texcoords),
                });
            }

            let indexed = !model.mesh.indices.is_empty();
            let textured = !model.mesh.texcoords.is_empty();
            let normal_source = if !model.mesh.normals.is_empty() {
                NormalSource::Provided(flat_to_v3(&model.mesh.normals))
            } else {
                NormalSource::ComputedFlat
            };

            let geometry = if indexed {
                Geometry::new_indexed(
                    flat_to_v3(&model.mesh.positions),
                    normal_source,
                    model.mesh.indices,
                    tan_space_info,
                )
            } else {
                Geometry::new_non_indexed(
                    flat_to_v3(&model.mesh.positions),
                    normal_source,
                    tan_space_info,
                )
            };

            let mut builder = MeshBuilder::new().with_geometry(geometry);

            if textured {
                builder = builder.with_texture_uvs(flat_to_v2(&model.mesh.texcoords));
            }

            if let Some(mat_idx) = model.mesh.material_id {
                let material = &materials[mat_idx].name;

                let material_id = local_materials
                    .iter()
                    .find(|(name, _)| name == material)
                    .map(|o| o.1)
                    .unwrap();

                mesh_materials.push(material_id);
            }

            meshes.push(builder.build()?);
        }

        Ok((meshes, mesh_materials))
    }
}
