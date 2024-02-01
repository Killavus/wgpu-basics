use nalgebra as na;
type FVec4 = na::Vector4<f32>;
type FVec3 = na::Vector3<f32>;
type FVec2 = na::Vector2<f32>;

use crate::mesh::{Geometry, NormalSource};

pub struct Plane;

impl Plane {
    pub fn new() -> Self {
        Self
    }

    pub fn geometry() -> Geometry {
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

        Geometry::new_indexed(mesh, NormalSource::Provided(normals.to_vec()), faces)
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

        Geometry::new_indexed(mesh, NormalSource::Provided(normals), faces)
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
        ]
    }
}
