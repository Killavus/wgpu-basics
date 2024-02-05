use anyhow::Result;
use nalgebra as na;
use std::path::Path;

use crate::{
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
    pub fn load(path: impl AsRef<Path>, _material_atlas: &mut MaterialAtlas) -> Result<Vec<Mesh>> {
        let (models, _materials) = tobj::load_obj(path.as_ref(), &tobj::LoadOptions::default())?;

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
