use anyhow::{Context, Result};
use nalgebra as na;
use std::path::Path;

use crate::{
    gpu::Gpu,
    material::MaterialAtlas,
    mesh::{Geometry, Mesh, MeshBuilder, NormalSource},
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

impl ObjLoader {
    pub fn load(
        path: impl AsRef<Path>,
        gpu: &Gpu,
        material_atlas: &mut MaterialAtlas,
    ) -> Result<Vec<Mesh>> {
        let (models, materials) = tobj::load_obj(path.as_ref(), &tobj::LoadOptions::default())
            .context("failed to load obj file")?;

        let materials = materials?;

        let mut local_materials = Vec::with_capacity(materials.len());

        for material in materials {
            let is_phong_solid = material.diffuse.is_some() && material.ambient.is_some();
            let is_phong_textured = material.diffuse_texture.is_some();

            if is_phong_solid {
                let ambient = material.ambient.unwrap();
                let ambient = na::Vector4::new(ambient[0], ambient[1], ambient[2], 0.0);
                let diffuse_f = material.diffuse.unwrap();
                let diffuse = na::Vector4::new(diffuse_f[0], diffuse_f[1], diffuse_f[2], 0.0);
                let specular = material.specular.unwrap_or(diffuse_f);
                let specular = na::Vector4::new(specular[0], specular[1], specular[2], 0.0);

                local_materials.push((
                    material.name.clone(),
                    material_atlas.add_phong_solid(gpu, ambient, diffuse, specular),
                ));
            } else if is_phong_textured {
                let diffuse_texture = material
                    .diffuse_texture
                    .map(|tex_path| {
                        let base_path = path.as_ref().parent().unwrap_or(path.as_ref());
                        base_path.join(tex_path)
                    })
                    .unwrap();

                let specular = material.specular_texture.map(|tex_path| {
                    let base_path = path.as_ref().parent().unwrap_or(path.as_ref());
                    base_path.join(tex_path)
                });

                local_materials.push((
                    material.name.clone(),
                    material_atlas.add_phong_textured(gpu, &diffuse_texture, specular.as_ref()),
                ));
            }
        }

        let mut meshes = vec![];
        for model in models {
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
                )
            } else {
                Geometry::new_non_indexed(flat_to_v3(&model.mesh.positions), normal_source)
            };

            let mut builder = MeshBuilder::new().with_geometry(geometry);

            if textured {
                builder = builder.with_texture_uvs(flat_to_v2(&model.mesh.texcoords));
            }

            meshes.push(builder.build()?);
        }

        Ok(meshes)
    }
}
