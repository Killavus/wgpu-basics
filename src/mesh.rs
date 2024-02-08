use anyhow::Result;
use nalgebra as na;
type FVec3 = na::Vector3<f32>;
type FVec2 = na::Vector2<f32>;

#[derive(Default)]
struct MeshVertexAttributes {
    texture: Option<TextureUV>,
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub enum MeshVertexArrayType {
    PN,
    PNUV,
    PNTBUV,
}

impl MeshVertexArrayType {
    pub fn stride(&self) -> usize {
        match self {
            Self::PN => PN_STRIDE,
            Self::PNUV => PNUV_STRIDE,
            Self::PNTBUV => PNTBUV_STRIDE,
        }
    }
}

impl MeshVertexAttributes {
    fn has_texture_uvs(&self) -> bool {
        self.texture.is_some()
    }

    pub fn vertex_array_type(&self) -> MeshVertexArrayType {
        if self.has_texture_uvs() {
            MeshVertexArrayType::PNUV
        } else {
            MeshVertexArrayType::PN
        }
    }
}

struct TextureUV {
    uv: Vec<FVec2>,
}

impl TextureUV {
    fn new(uv: Vec<FVec2>) -> Self {
        Self { uv }
    }
}

pub struct Mesh {
    geometry: Geometry,
    vertex_attributes: MeshVertexAttributes,
}

impl Mesh {
    const PN_VERTEX_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        step_mode: wgpu::VertexStepMode::Vertex,
        array_stride: PN_STRIDE as wgpu::BufferAddress,
        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
    };

    const PNUV_VERTEX_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        step_mode: wgpu::VertexStepMode::Vertex,
        array_stride: PNUV_STRIDE as wgpu::BufferAddress,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3,
            2 => Float32x2,
        ],
    };

    const PNTBUV_VERTEX_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        step_mode: wgpu::VertexStepMode::Vertex,
        array_stride: PNTBUV_STRIDE as wgpu::BufferAddress,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3,
            2 => Float32x3,
            3 => Float32x3,
            4 => Float32x2,
        ],
    };

    pub fn pntbuv_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
        Self::PNTBUV_VERTEX_LAYOUT
    }

    pub fn pnuv_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
        Self::PNUV_VERTEX_LAYOUT
    }

    pub fn pn_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
        Self::PN_VERTEX_LAYOUT
    }

    pub fn vertex_array_type(&self) -> MeshVertexArrayType {
        let attr_vertex_type = self.vertex_attributes.vertex_array_type();

        match attr_vertex_type {
            MeshVertexArrayType::PN => MeshVertexArrayType::PN,
            MeshVertexArrayType::PNUV => {
                if self.geometry.has_tangent_space() {
                    MeshVertexArrayType::PNTBUV
                } else {
                    MeshVertexArrayType::PNUV
                }
            }
            _ => attr_vertex_type,
        }
    }

    pub fn is_indexed(&self) -> bool {
        match &self.geometry {
            Geometry::Indexed { .. } => true,
            Geometry::NonIndexed { .. } => false,
        }
    }

    pub fn copy_to_index_buffer(&self, index_buffer: &mut Vec<u32>) {
        let faces = match &self.geometry {
            Geometry::Indexed { faces, .. } => faces,
            Geometry::NonIndexed { .. } => return,
        };

        index_buffer.reserve(faces.len());
        index_buffer.extend_from_slice(faces);
    }

    pub fn num_vertices(&self) -> usize {
        self.geometry.vertex_count()
    }

    pub fn num_indices(&self) -> Option<usize> {
        match &self.geometry {
            Geometry::Indexed { faces, .. } => Some(faces.len()),
            Geometry::NonIndexed { .. } => None,
        }
    }

    pub fn copy_to_mesh_bank(&self, vertex_array: &mut Vec<u8>) {
        let vertex_count = self.geometry.vertex_count();
        let mesh_size = match self.vertex_array_type() {
            MeshVertexArrayType::PNUV => vertex_count * PNUV_STRIDE,
            MeshVertexArrayType::PN => vertex_count * PN_STRIDE,
            MeshVertexArrayType::PNTBUV => vertex_count * PNTBUV_STRIDE,
        };

        vertex_array.reserve(mesh_size);

        let mesh = match &self.geometry {
            Geometry::Indexed { mesh, .. } => mesh,
            Geometry::NonIndexed { mesh, .. } => mesh,
        };

        let normals = match &self.geometry {
            Geometry::Indexed { normals, .. } => normals,
            Geometry::NonIndexed { normals, .. } => normals,
        };

        for i in 0..vertex_count {
            let vertex = mesh[i];
            match normals {
                NormalInformation::ModelNormals(normals) => {
                    let normal = normals[i];

                    vertex_array.extend_from_slice(bytemuck::cast_slice(&[vertex]));
                    vertex_array.extend_from_slice(bytemuck::cast_slice(&[normal]));
                }
                NormalInformation::TangentSpace(normals, t_vectors, bt_vectors) => {
                    let normal = normals[i];
                    let t_vector = t_vectors[i];
                    let bt_vector = bt_vectors[i];

                    vertex_array.extend_from_slice(bytemuck::cast_slice(&[vertex]));
                    vertex_array.extend_from_slice(bytemuck::cast_slice(&[normal]));
                    vertex_array.extend_from_slice(bytemuck::cast_slice(&[t_vector]));
                    vertex_array.extend_from_slice(bytemuck::cast_slice(&[bt_vector]));
                }
            }

            if let Some(texture) = &self.vertex_attributes.texture {
                vertex_array.extend_from_slice(bytemuck::cast_slice(&[texture.uv[i]]));
            }
        }
    }
}

pub struct MeshBuilder {
    geometry: Option<Geometry>,
    vertex_attributes: MeshVertexAttributes,
}

pub const PNTBUV_STRIDE: usize = std::mem::size_of::<FVec3>() * 4 + std::mem::size_of::<FVec2>();
pub const PNUV_STRIDE: usize = std::mem::size_of::<FVec3>() * 2 + std::mem::size_of::<FVec2>();
pub const PN_STRIDE: usize = std::mem::size_of::<FVec3>() * 2;
pub const PNUV_SLOTS: u32 = 3;
pub const PN_SLOTS: u32 = 2;
pub const PNTBUV_SLOTS: u32 = 5;

impl MeshBuilder {
    pub fn new() -> Self {
        Self {
            geometry: None,
            vertex_attributes: MeshVertexAttributes::default(),
        }
    }

    pub fn with_geometry(mut self, geometry: Geometry) -> Self {
        self.geometry = Some(geometry);
        self
    }

    pub fn with_texture_uvs(mut self, texture_uvs: Vec<FVec2>) -> Self {
        self.vertex_attributes.texture = Some(TextureUV::new(texture_uvs));
        self
    }

    pub fn build(self) -> Result<Mesh> {
        Ok(Mesh {
            geometry: self
                .geometry
                .ok_or_else(|| anyhow::anyhow!("Mesh geometry not provided"))?,
            vertex_attributes: self.vertex_attributes,
        })
    }
}

#[derive(Debug)]
enum NormalInformation {
    ModelNormals(Vec<FVec3>),
    TangentSpace(Vec<FVec3>, Vec<FVec3>, Vec<FVec3>),
}

#[derive(Debug)]
pub enum Geometry {
    Indexed {
        mesh: Vec<FVec3>,
        normals: NormalInformation,
        faces: Vec<u32>,
    },
    NonIndexed {
        mesh: Vec<FVec3>,
        normals: NormalInformation,
    },
}

pub enum NormalSource {
    Provided(Vec<FVec3>),
    ComputedFlat,
}

pub struct TangentSpaceInformation {
    pub texture_uvs: Vec<FVec2>,
}

impl NormalSource {
    fn into_normals(
        self,
        mesh: &[FVec3],
        faces_iter: impl Iterator<Item = usize> + Clone,
        tangent_space_information: Option<TangentSpaceInformation>,
    ) -> NormalInformation {
        let normals = match self {
            Self::Provided(normals) => normals,
            Self::ComputedFlat => flat_normals(mesh, faces_iter.clone()),
        };

        match tangent_space_information {
            Some(TangentSpaceInformation { texture_uvs }) => {
                let (t_vectors, bt_vectors) = tangent_space_vectors(mesh, &texture_uvs, faces_iter);

                NormalInformation::TangentSpace(normals, t_vectors, bt_vectors)
            }
            None => NormalInformation::ModelNormals(normals),
        }
    }
}

impl Geometry {
    pub fn new_non_indexed(
        mesh: Vec<FVec3>,
        normals: NormalSource,
        tangent_space_information: Option<TangentSpaceInformation>,
    ) -> Self {
        let normals = normals.into_normals(&mesh, 0..mesh.len(), tangent_space_information);

        Self::NonIndexed { mesh, normals }
    }

    pub fn new_indexed(
        mesh: Vec<FVec3>,
        normals: NormalSource,
        faces: Vec<u32>,
        tangent_space_information: Option<TangentSpaceInformation>,
    ) -> Self {
        let normals = normals.into_normals(
            &mesh,
            faces.iter().copied().map(|idx| idx as usize),
            tangent_space_information,
        );

        Self::Indexed {
            mesh,
            normals,
            faces,
        }
    }

    pub fn has_tangent_space(&self) -> bool {
        match self {
            Geometry::Indexed { normals, .. } => match normals {
                NormalInformation::TangentSpace(_, _, _) => true,
                _ => false,
            },
            Geometry::NonIndexed { normals, .. } => match normals {
                NormalInformation::TangentSpace(_, _, _) => true,
                _ => false,
            },
        }
    }

    pub fn vertex_count(&self) -> usize {
        match self {
            Geometry::Indexed { mesh, .. } => mesh.len(),
            Geometry::NonIndexed { mesh, .. } => mesh.len(),
        }
    }
}

fn flat_normals(mesh: &[FVec3], mut idx_iter: impl Iterator<Item = usize>) -> Vec<FVec3> {
    let mut normals = vec![FVec3::zeros(); mesh.len()];

    loop {
        let triangle_idx = idx_iter
            .next()
            .zip(idx_iter.next())
            .zip(idx_iter.next())
            .map(|((i0, i1), i2)| (i0, i1, i2));

        match triangle_idx {
            Some((i0, i1, i2)) => {
                let v0 = mesh[i0];
                let v1 = mesh[i1];
                let v2 = mesh[i2];

                let e1 = v1 - v0;
                let e2 = v2 - v0;

                let normal = e1.cross(&e2).normalize();
                normals[i0] += normal;
                normals[i0] = normals[i0].normalize();
                normals[i1] += normal;
                normals[i1] = normals[i1].normalize();
                normals[i2] += normal;
                normals[i2] = normals[i2].normalize();
            }
            None => {
                break;
            }
        }
    }

    normals
}

fn tangent_space_vectors(
    mesh: &[FVec3],
    texture_uvs: &[FVec2],
    mut idx_iter: impl Iterator<Item = usize>,
) -> (Vec<FVec3>, Vec<FVec3>) {
    let mut t_vectors = vec![na::Vector3::zeros(); mesh.len()];
    let mut bt_vectors = vec![na::Vector3::zeros(); mesh.len()];

    loop {
        let triangle_idx = idx_iter
            .next()
            .zip(idx_iter.next())
            .zip(idx_iter.next())
            .map(|((i0, i1), i2)| (i0, i1, i2));

        match triangle_idx {
            Some((i0, i1, i2)) => {
                for (i, (j, k)) in [(i0, (i1, i2)), (i1, (i0, i2)), (i2, (i0, i1))] {
                    let v0 = mesh[i];
                    let v1 = mesh[j];
                    let v2 = mesh[k];

                    let e1 = v1 - v0;
                    let e2 = v2 - v0;

                    let delta_uv1 = texture_uvs[j] - texture_uvs[i];
                    let delta_uv2 = texture_uvs[k] - texture_uvs[i];

                    let det = 1.0 / (delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x);

                    let mut tangent = FVec3::zeros();
                    let mut bitangent = FVec3::zeros();

                    tangent.x = det * (delta_uv2.y * e1.x - delta_uv1.y * e2.x);
                    tangent.y = det * (delta_uv2.y * e1.y - delta_uv1.y * e2.y);
                    tangent.z = det * (delta_uv2.y * e1.z - delta_uv1.y * e2.z);

                    bitangent.x = det * (-delta_uv2.x * e1.x + delta_uv1.x * e2.x);
                    bitangent.y = det * (-delta_uv2.x * e1.y + delta_uv1.x * e2.y);
                    bitangent.z = det * (-delta_uv2.x * e1.z + delta_uv1.x * e2.z);

                    t_vectors[i] += tangent;
                    t_vectors[i] = t_vectors[i].normalize();
                    bt_vectors[i] += bitangent;
                    bt_vectors[i] = bt_vectors[i].normalize();
                }
            }
            None => {
                break;
            }
        }
    }

    (t_vectors, bt_vectors)
}
