use anyhow::Result;
use nalgebra as na;
use std::path::Path;
type FVec3 = na::Vector3<f32>;

#[derive(Debug)]
pub enum Model {
    Indexed {
        mesh: Vec<FVec3>,
        normals: Vec<FVec3>,
        faces: Vec<u32>,
    },
    NonIndexed {
        mesh: Vec<FVec3>,
        normals: Vec<FVec3>,
    },
}

#[derive(Default)]
struct ModelBuilder {
    mesh: Vec<FVec3>,
    faces: Option<Vec<u32>>,
    normals: Option<Vec<FVec3>>,
}

impl ModelBuilder {
    fn new(mesh: impl IntoIterator<Item = FVec3>) -> Self {
        Self {
            mesh: mesh.into_iter().collect(),
            ..Default::default()
        }
    }

    fn with_faces(mut self, faces: impl IntoIterator<Item = u32>) -> Self {
        self.faces = Some(faces.into_iter().collect());
        self
    }

    fn with_normals(mut self, normals: impl IntoIterator<Item = FVec3>) -> Self {
        self.normals = Some(normals.into_iter().collect());
        self
    }

    fn build(self) -> Model {
        let Self {
            faces,
            mesh,
            mut normals,
        } = self;

        if normals.is_none() {
            match &faces {
                Some(faces) => {
                    normals = Some(Self::flat_normals(
                        &mesh,
                        faces.iter().copied().map(|idx| idx as usize),
                    ));
                }
                None => {
                    normals = Some(Self::flat_normals(&mesh, 0..mesh.len()));
                }
            }
        }

        if let Some(faces) = faces {
            Model::new_indexed(mesh, faces, normals.unwrap())
        } else {
            Model::new(mesh, normals.unwrap())
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
}

pub struct GpuModel {
    model: Model,
    vertex_buf: wgpu::Buffer,
    index_buf: Option<wgpu::Buffer>,
}

impl GpuModel {
    const VERTEX_ATTRS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
    ];

    pub const VERTEX_ATTRS_MAX_SLOT: u32 = 2;

    pub fn new(device: &wgpu::Device, model: Model) -> Self {
        use wgpu::util::DeviceExt;

        let gpu_vertices = model
            .mesh()
            .iter()
            .copied()
            .zip(model.normals().iter().copied())
            .flat_map(|(v, n)| [v, n])
            .collect::<Vec<_>>();

        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(gpu_vertices.as_slice()),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buf = model.faces().map(|faces| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(faces),
                usage: wgpu::BufferUsages::INDEX,
            })
        });

        Self {
            model,
            vertex_buf,
            index_buf,
        }
    }

    pub fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: (std::mem::size_of::<FVec3>() * 2) as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::VERTEX_ATTRS,
        }
    }

    pub fn num_indices(&self) -> u32 {
        self.model.num_indices()
    }

    pub fn indexed(&self) -> bool {
        self.index_buf.is_some()
    }

    pub fn configure_pass<'rpass, 'model: 'rpass>(
        &'model self,
        render_pass: &mut wgpu::RenderPass<'rpass>,
    ) {
        render_pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        if let Some(index_buf) = &self.index_buf {
            render_pass.set_index_buffer(index_buf.slice(..), wgpu::IndexFormat::Uint32);
        }
    }
}

impl Model {
    fn new_indexed(mesh: Vec<FVec3>, faces: Vec<u32>, normals: Vec<FVec3>) -> Self {
        Self::Indexed {
            mesh,
            normals,
            faces,
        }
    }

    fn new(mesh: Vec<FVec3>, normals: Vec<FVec3>) -> Self {
        Self::NonIndexed { mesh, normals }
    }

    fn mesh(&self) -> &[FVec3] {
        match self {
            Self::Indexed { mesh, .. } => mesh,
            Self::NonIndexed { mesh, .. } => mesh,
        }
    }

    fn normals(&self) -> &[FVec3] {
        match self {
            Self::Indexed { normals, .. } => normals,
            Self::NonIndexed { normals, .. } => normals,
        }
    }

    fn faces(&self) -> Option<&[u32]> {
        match self {
            Self::Indexed { faces, .. } => Some(faces),
            Self::NonIndexed { .. } => None,
        }
    }

    pub fn num_indices(&self) -> u32 {
        match self {
            Self::Indexed { faces, .. } => faces.len() as u32,
            Self::NonIndexed { mesh, .. } => mesh.len() as u32,
        }
    }
}

pub struct ObjParser;

impl ObjParser {
    pub fn read_model(path: impl AsRef<Path>) -> Result<Model> {
        use std::fs::File;
        use std::io::{prelude::*, BufReader};

        let reader = BufReader::new(File::open(path)?);
        let mut vertices = vec![];
        let mut has_faces = false;
        let mut faces = vec![];
        let mut has_normals = false;
        let mut normals = vec![];

        for line in reader.lines() {
            let line = line?;

            if line.is_empty() {
                continue;
            }

            match &line[0..1] {
                "v" => {
                    let mut iter = line.split_whitespace();
                    iter.next();
                    let x = iter.next().unwrap().parse::<f32>().unwrap();
                    let y = iter.next().unwrap().parse::<f32>().unwrap();
                    let z = iter.next().unwrap().parse::<f32>().unwrap();
                    vertices.push(FVec3::new(x, y, z));
                }
                "n" => {
                    has_normals = true;
                    let mut iter = line.split_whitespace();
                    iter.next();
                    let x = iter.next().unwrap().parse::<f32>().unwrap();
                    let y = iter.next().unwrap().parse::<f32>().unwrap();
                    let z = iter.next().unwrap().parse::<f32>().unwrap();
                    normals.push(FVec3::new(x, y, z));
                }
                "f" => {
                    has_faces = true;

                    let mut iter = line.split_whitespace();
                    iter.next();
                    let x = iter.next().unwrap().parse::<u32>().unwrap();
                    let y = iter.next().unwrap().parse::<u32>().unwrap();
                    let z = iter.next().unwrap().parse::<u32>().unwrap();
                    faces.push(x - 1);
                    faces.push(y - 1);
                    faces.push(z - 1);
                }
                _ => {}
            }
        }

        let mut builder = ModelBuilder::new(vertices);
        if has_faces {
            builder = builder.with_faces(faces);
        }

        if has_normals {
            builder = builder.with_normals(normals);
        }

        Ok(builder.build())
    }
}

pub struct Plane;

impl Plane {
    pub fn new() -> Self {
        Self
    }

    pub fn model(&self) -> Model {
        let mut mesh = vec![];
        let mut faces = vec![];

        let normal = na::Vector3::<f32>::y();
        let normals = [normal; 4];
        let x = na::Vector3::<f32>::x();
        let z = na::Vector3::<f32>::z();

        let tl = -0.5 * x - 0.5 * z;
        let tr = 0.5 * x - 0.5 * z;
        let bl = -0.5 * x + 0.5 * z;
        let br = 0.5 * x + 0.5 * z;

        mesh.push(tl);
        mesh.push(tr);
        mesh.push(bl);
        mesh.push(br);

        faces.push(2);
        faces.push(3);
        faces.push(0);
        faces.push(0);
        faces.push(3);
        faces.push(1);

        ModelBuilder::new(mesh)
            .with_faces(faces)
            .with_normals(normals)
            .build()
    }
}

pub struct Cube;

impl Cube {
    pub fn new() -> Self {
        Self
    }

    pub fn model(&self) -> Model {
        let center = FVec3::zeros();
        let mut mesh = vec![];
        let mut faces = vec![];

        let half_size = 0.5;
        mesh.push(FVec3::new(-half_size, half_size, half_size)); // front-tl 0
        mesh.push(FVec3::new(half_size, half_size, half_size)); // front-tr 1
        mesh.push(FVec3::new(-half_size, -half_size, half_size)); // front-bl 2
        mesh.push(FVec3::new(half_size, -half_size, half_size)); // front-br 3
        mesh.push(FVec3::new(-half_size, half_size, -half_size)); // back-tl 4
        mesh.push(FVec3::new(half_size, half_size, -half_size)); // back-tr 5
        mesh.push(FVec3::new(-half_size, -half_size, -half_size)); // back-bl 6
        mesh.push(FVec3::new(half_size, -half_size, -half_size)); // back-br 7

        faces.push(2);
        faces.push(1);
        faces.push(0);
        faces.push(1);
        faces.push(2);
        faces.push(3);

        faces.push(4);
        faces.push(5);
        faces.push(6);
        faces.push(7);
        faces.push(6);
        faces.push(5);

        faces.push(0);
        faces.push(1);
        faces.push(4);
        faces.push(5);
        faces.push(4);
        faces.push(1);

        faces.push(6);
        faces.push(3);
        faces.push(2);
        faces.push(6);
        faces.push(7);
        faces.push(3);

        faces.push(4);
        faces.push(2);
        faces.push(0);
        faces.push(4);
        faces.push(6);
        faces.push(2);

        faces.push(7);
        faces.push(5);
        faces.push(1);
        faces.push(1);
        faces.push(3);
        faces.push(7);

        let normals: Vec<FVec3> = mesh
            .iter()
            .copied()
            .map(|v| (v - center).normalize())
            .collect();

        ModelBuilder::new(mesh)
            .with_faces(faces)
            .with_normals(normals)
            .build()
    }
}
