use nalgebra as na;
type FVec4 = na::Vector4<f32>;
type FVec3 = na::Vector3<f32>;
type FVec2 = na::Vector2<f32>;

use crate::mesh::{Geometry, NormalSource, TangentSpaceInformation};

pub struct UVSphere;

impl UVSphere {
    pub fn geometry(slices: usize, stacks: usize) -> Geometry {
        let stack_angle = std::f32::consts::PI / stacks as f32;
        let slice_angle = 2.0 * std::f32::consts::PI / slices as f32;

        let mut mesh = vec![na::Vector3::new(0.0, 1.0, 0.0)];

        for i in 0..(stacks - 1) {
            let angle = (i + 1) as f32 * stack_angle;
            let y = angle.cos();
            let r = angle.sin();

            for i in 0..slices {
                let angle = i as f32 * slice_angle;
                let x = r * angle.cos();
                let z = r * angle.sin();

                mesh.push(na::Vector3::new(x, y, z));
            }
        }

        mesh.push(na::Vector3::new(0.0, -1.0, 0.0));

        let mut faces: Vec<u32> = vec![];
        let top_vert = 0;
        let bottom_vert = (mesh.len() - 1) as u32;

        for i in 0..slices {
            let i0 = i + 1;
            let i1 = (i + 1) % slices + 1;

            faces.push(top_vert);
            faces.push(i1 as u32);
            faces.push(i0 as u32);

            faces.push(bottom_vert);
            faces.push(bottom_vert - i1 as u32);
            faces.push(bottom_vert - i0 as u32);
        }

        for i in 1..(stacks - 1) {
            for j in 0..slices {
                let t0 = (i - 1) * slices + j + 1;
                let t1 = (i - 1) * slices + (j + 1) % slices + 1;
                let b0 = i * slices + j + 1;
                let b1 = i * slices + (j + 1) % slices + 1;

                faces.push(t0 as u32);
                faces.push(b1 as u32);
                faces.push(b0 as u32);
                faces.push(b1 as u32);
                faces.push(t0 as u32);
                faces.push(t1 as u32);
            }
        }

        let normals = mesh.iter().map(|v| v.normalize()).collect::<Vec<_>>();

        Geometry::new_indexed(mesh, NormalSource::Provided(normals), faces, None)
    }
}

pub struct Plane;

impl Plane {
    pub(self) fn raw_geometry() -> (Vec<na::Vector3<f32>>, Vec<na::Vector3<f32>>, Vec<u32>) {
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

        (mesh, normals.to_vec(), faces)
    }

    pub fn geometry_tan_space() -> Geometry {
        let (mesh, normals, faces) = Self::raw_geometry();

        Geometry::new_indexed(
            mesh,
            NormalSource::Provided(normals.to_vec()),
            faces,
            Some(TangentSpaceInformation {
                texture_uvs: Self::uvs(),
            }),
        )
    }

    pub fn geometry() -> Geometry {
        let (mesh, normals, faces) = Self::raw_geometry();

        Geometry::new_indexed(mesh, NormalSource::Provided(normals.to_vec()), faces, None)
    }

    pub fn uvs() -> Vec<FVec2> {
        vec![
            FVec2::new(0.0, 0.0),
            FVec2::new(1.0, 0.0),
            FVec2::new(0.0, 1.0),
            FVec2::new(1.0, 1.0),
        ]
    }
}

pub struct Cube;

impl Cube {
    pub fn geometry() -> Geometry {
        let (mesh, normals, faces) = Self::raw_geometry();

        Geometry::new_indexed(mesh, NormalSource::Provided(normals), faces, None)
    }

    pub fn geometry_tan_space() -> Geometry {
        let (mesh, normals, faces) = Self::raw_geometry();

        Geometry::new_indexed(
            mesh,
            NormalSource::Provided(normals),
            faces,
            Some(TangentSpaceInformation {
                texture_uvs: Self::uvs(),
            }),
        )
    }

    fn raw_geometry() -> (Vec<FVec3>, Vec<FVec3>, Vec<u32>) {
        let (face_v, face_normals, face_indexes) = Plane::raw_geometry();
        let half_size = 0.5;

        let mut mesh = Vec::with_capacity(24);
        let mut normals = Vec::with_capacity(24);
        let mut faces = Vec::with_capacity(36);

        let (top_face, top_normals) = (
            face_v
                .iter()
                .map(|v| v + na::Vector3::new(0.0, half_size, 0.0)),
            face_normals.iter().copied(),
        );

        let (bottom_face, bottom_normals) = (
            face_v.iter().map(|v| {
                na::Matrix4::new_rotation(na::Vector3::x() * 180.0f32.to_radians())
                    .transform_vector(v)
                    + na::Vector3::new(0.0, -half_size, 0.0)
            }),
            face_normals.iter().map(|n| {
                na::Matrix4::new_rotation(na::Vector3::x() * 180.0f32.to_radians())
                    .transform_vector(n)
            }),
        );

        let (front_face, front_normals) = (
            face_v.iter().map(|v| {
                na::Matrix4::new_rotation(na::Vector3::x() * 90.0f32.to_radians())
                    .transform_vector(v)
                    + na::Vector3::new(0.0, 0.0, half_size)
            }),
            face_normals.iter().map(|n| {
                na::Matrix4::new_rotation(na::Vector3::x() * 90.0f32.to_radians())
                    .transform_vector(n)
            }),
        );

        let (back_face, back_normals) = (
            face_v.iter().map(|v| {
                na::Matrix4::new_rotation(na::Vector3::x() * -90.0f32.to_radians())
                    .transform_vector(v)
                    + na::Vector3::new(0.0, 0.0, -half_size)
            }),
            face_normals.iter().map(|n| {
                na::Matrix4::new_rotation(na::Vector3::x() * -90.0f32.to_radians())
                    .transform_vector(n)
            }),
        );

        let (left_face, left_normals) = (
            face_v.iter().map(|v| {
                (na::Matrix4::new_rotation(na::Vector3::x() * 90.0f32.to_radians())
                    * na::Matrix4::new_rotation(na::Vector3::z() * 90.0f32.to_radians()))
                .transform_vector(v)
                    + na::Vector3::new(-half_size, 0.0, 0.0)
            }),
            face_normals.iter().map(|n| {
                (na::Matrix4::new_rotation(na::Vector3::x() * 90.0f32.to_radians())
                    * na::Matrix4::new_rotation(na::Vector3::z() * 90.0f32.to_radians()))
                .transform_vector(n)
            }),
        );

        let (right_face, right_normals) = (
            face_v.iter().map(|v| {
                (na::Matrix4::new_rotation(na::Vector3::x() * -90.0f32.to_radians())
                    * na::Matrix4::new_rotation(na::Vector3::z() * -90.0f32.to_radians()))
                .transform_vector(v)
                    + na::Vector3::new(half_size, 0.0, 0.0)
            }),
            face_normals.iter().map(|n| {
                (na::Matrix4::new_rotation(na::Vector3::x() * -90.0f32.to_radians())
                    * na::Matrix4::new_rotation(na::Vector3::z() * -90.0f32.to_radians()))
                .transform_vector(n)
            }),
        );

        mesh.extend(top_face);
        mesh.extend(bottom_face);
        mesh.extend(front_face);
        mesh.extend(back_face);
        mesh.extend(left_face);
        mesh.extend(right_face);

        normals.extend(top_normals);
        normals.extend(bottom_normals);
        normals.extend(front_normals);
        normals.extend(back_normals);
        normals.extend(left_normals);
        normals.extend(right_normals);

        faces.extend(face_indexes.iter());
        faces.extend(face_indexes.iter().map(|i| i + 4));
        faces.extend(face_indexes.iter().map(|i| i + 8));
        faces.extend(face_indexes.iter().map(|i| i + 12));
        faces.extend(face_indexes.iter().map(|i| i + 16));
        faces.extend(face_indexes.iter().map(|i| i + 20));

        (mesh, normals, faces)
    }

    pub fn uvs() -> Vec<FVec2> {
        vec![
            FVec2::new(0.0, 0.0),
            FVec2::new(1.0, 0.0),
            FVec2::new(0.0, 1.0),
            FVec2::new(1.0, 1.0),
            FVec2::new(0.0, 0.0),
            FVec2::new(1.0, 0.0),
            FVec2::new(0.0, 1.0),
            FVec2::new(1.0, 1.0),
            FVec2::new(0.0, 0.0),
            FVec2::new(1.0, 0.0),
            FVec2::new(0.0, 1.0),
            FVec2::new(1.0, 1.0),
            FVec2::new(0.0, 0.0),
            FVec2::new(1.0, 0.0),
            FVec2::new(0.0, 1.0),
            FVec2::new(1.0, 1.0),
            FVec2::new(0.0, 0.0),
            FVec2::new(1.0, 0.0),
            FVec2::new(0.0, 1.0),
            FVec2::new(1.0, 1.0),
            FVec2::new(0.0, 0.0),
            FVec2::new(1.0, 0.0),
            FVec2::new(0.0, 1.0),
            FVec2::new(1.0, 1.0),
        ]
    }
}
