use crate::{
    camera::{Camera, GpuCamera},
    gpu::Gpu,
    loader::{ObjLoader, ObjLoaderSettings},
    material::{MaterialAtlas, SpecularTexture},
    mesh::MeshBuilder,
    light_scene::LightScene,
    projection::{wgpu_projection, GpuProjection},
    scene::{Instance, Scene, SceneModelBuilder, SceneObjectId},
    shapes::{Cube, Plane, UVSphere},
};
use anyhow::Result;
use image::EncodableLayout;
use nalgebra as na;
use std::collections::HashMap;

type TestScene = (
    Scene,
    MaterialAtlas,
    LightScene,
    GpuCamera,
    GpuProjection,
    na::Matrix4<f32>,
    HashMap<String, SceneObjectId>,
);

pub fn load_skybox(gpu: &Gpu) -> Result<wgpu::Texture> {
    let (sky_width, sky_height, sky_data) = [
        image::open("./textures/skybox/posx.jpg")?,
        image::open("./textures/skybox/negx.jpg")?,
        image::open("./textures/skybox/posy.jpg")?,
        image::open("./textures/skybox/negy.jpg")?,
        image::open("./textures/skybox/posz.jpg")?,
        image::open("./textures/skybox/negz.jpg")?,
    ]
    .into_iter()
    .fold((0, 0, vec![]), |mut acc, img| {
        acc.0 = img.width();
        acc.1 = img.height();
        acc.2.extend_from_slice(img.to_rgba8().as_bytes());

        acc
    });

    let skybox_size = wgpu::Extent3d {
        width: sky_width,
        height: sky_height,
        depth_or_array_layers: 6,
    };

    let skybox_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: skybox_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    gpu.queue.write_texture(
        skybox_tex.as_image_copy(),
        &sky_data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * sky_width),
            rows_per_image: Some(sky_height),
        },
        skybox_size,
    );

    Ok(skybox_tex)
}

pub fn blinn_phong_scene(gpu: &Gpu) -> Result<TestScene> {
    let mut scene = Scene::default();
    let mut material_atlas = MaterialAtlas::new(gpu);

    let plane_uv = MeshBuilder::new()
        .with_geometry(Plane::geometry())
        .with_texture_uvs(Plane::uvs().into_iter().map(|uv| uv * 10.0).collect())
        .build()?;

    let plane_uv = scene.load_model(SceneModelBuilder::default().with_meshes(vec![plane_uv]));
    let woodfloor = material_atlas.add_phong_textured(
        gpu,
        "./textures/woodfloor_detail.jpg",
        SpecularTexture::Ideal(64.0),
    )?;

    scene.add_object_with_material(
        plane_uv,
        Instance::new_model(na::Matrix4::new_scaling(100.0)),
        woodfloor,
    );

    let projection_mat =
        na::Matrix4::new_perspective(gpu.aspect_ratio(), 45.0f32.to_radians(), 0.1, 100.0);

    let projection: GpuProjection = GpuProjection::new(projection_mat, &gpu.device)?;
    let projection_mat = wgpu_projection(projection_mat);

    let mut lights = LightScene::default();

    lights.new_point(
        na::Vector3::new(0.0, 3.0, 0.0),
        na::Vector3::new(0.05, 0.05, 0.05),
        na::Vector3::new(1.0, 1.0, 1.0),
        na::Vector3::new(0.3, 0.3, 0.3),
        na::Vector3::new(1.0, 0.09, 0.0018),
    );

    let mut camera = GpuCamera::new(
        Camera::new(
            na::Point3::new(0.0, 18.0, 14.0),
            -45.0f32.to_radians(),
            270.0f32.to_radians(),
        ),
        &gpu.device,
    )?;

    Ok((
        scene,
        material_atlas,
        lights,
        camera,
        projection,
        wgpu_projection(projection_mat),
        HashMap::default(),
    ))
}

pub fn teapot_scene(gpu: &Gpu) -> Result<TestScene> {
    let mut scene = Scene::default();
    let mut material_atlas = MaterialAtlas::new(gpu);

    let cube_mesh = MeshBuilder::new().with_geometry(Cube::geometry()).build()?;
    let cube_uvtb_mesh = MeshBuilder::new()
        .with_geometry(Cube::geometry_tan_space())
        .with_texture_uvs(Cube::uvs())
        .build()?;

    let plane_mesh = MeshBuilder::new()
        .with_geometry(Plane::geometry())
        .build()?;

    let sphere_mesh = MeshBuilder::new()
        .with_geometry(UVSphere::geometry(32, 32))
        .build()?;

    let (teapot_mesh, _) = ObjLoader::load(
        "./models/teapot.obj",
        gpu,
        &mut material_atlas,
        ObjLoaderSettings {
            calculate_tangent_space: false,
        },
    )?;

    let (maya_mesh, maya_materials) = ObjLoader::load(
        "./models/maya/maya.obj",
        gpu,
        &mut material_atlas,
        ObjLoaderSettings {
            calculate_tangent_space: true,
        },
    )?;

    let teapot = scene.load_model(SceneModelBuilder::default().with_meshes(teapot_mesh));
    let cube = scene.load_model(SceneModelBuilder::default().with_meshes(vec![cube_mesh]));
    let plane = scene.load_model(SceneModelBuilder::default().with_meshes(vec![plane_mesh]));
    let uv_sphere = scene.load_model(SceneModelBuilder::default().with_meshes(vec![sphere_mesh]));

    let cube_uv_nmap =
        scene.load_model(SceneModelBuilder::default().with_meshes(vec![cube_uvtb_mesh]));

    let maya = scene.load_model(
        SceneModelBuilder::default()
            .with_meshes(maya_mesh)
            .with_local_materials(maya_materials),
    );

    let light_gray = material_atlas.add_phong_solid(
        gpu,
        na::Vector4::new(0.6, 0.6, 0.6, 0.1),
        na::Vector4::new(0.6, 0.6, 0.6, 0.7),
        na::Vector4::new(0.6, 0.6, 0.6, 64.0),
    )?;

    let lily = material_atlas.add_phong_solid(
        gpu,
        na::Vector4::new(0.5, 0.5, 1.0, 0.0),
        na::Vector4::new(0.5, 0.5, 1.0, 0.0),
        na::Vector4::new(0.5, 0.5, 1.0, 32.0),
    )?;

    let quite_red = material_atlas.add_phong_solid(
        gpu,
        na::Vector4::new(0.8, 0.2, 0.2, 0.1),
        na::Vector4::new(0.8, 0.2, 0.2, 0.7),
        na::Vector4::new(0.8, 0.2, 0.2, 16.0),
    )?;

    let white = material_atlas.add_phong_solid(
        gpu,
        na::Vector4::new(1.0, 1.0, 1.0, 0.1),
        na::Vector4::new(1.0, 1.0, 1.0, 0.7),
        na::Vector4::new(1.0, 1.0, 1.0, 64.0),
    )?;

    let toxic_green = material_atlas.add_phong_solid(
        gpu,
        na::Vector4::new(0.2, 0.8, 0.4, 0.0),
        na::Vector4::new(0.2, 0.8, 0.4, 0.0),
        na::Vector4::new(0.2, 0.2, 0.4, 64.0),
    )?;

    let brickwall_nmap = material_atlas.add_phong_textured_normal(
        gpu,
        "./textures/brickwall_diffuse.jpg",
        SpecularTexture::Ideal(32.0),
        "./textures/brickwall_normal.jpg",
    )?;

    scene.add_object_with_material(
        cube,
        Instance::new_model(
            na::Matrix4::new_translation(&na::Vector3::new(4.0, 4.5, -2.0))
                * na::Matrix4::new_rotation(na::Vector3::y() * 45.0f32.to_radians())
                * na::Matrix4::new_scaling(1.0),
        ),
        quite_red,
    );

    scene.add_object_with_material(
        uv_sphere,
        Instance::new_model(na::Matrix4::new_translation(&na::Vector3::new(
            5.0, 2.0, -4.0,
        ))),
        quite_red,
    );

    scene.add_object_with_material(
        cube,
        Instance::new_model(
            na::Matrix4::new_translation(&na::Vector3::new(12.0, 12.0, 0.0))
                * na::Matrix4::new_scaling(0.5),
        ),
        white,
    );

    scene.add_object_with_material(
        cube_uv_nmap,
        Instance::new_model(
            na::Matrix4::new_translation(&na::Vector3::new(1.0, 0.5, 1.0))
                * na::Matrix4::new_nonuniform_scaling(&na::Vector3::new(1.0, 2.0, 1.0)),
        ),
        brickwall_nmap,
    );

    scene.add_object_with_material(
        cube,
        Instance::new_model(na::Matrix4::new_translation(&na::Vector3::new(
            -6.0, 0.5, -4.0,
        ))),
        toxic_green,
    );

    scene.add_object_with_material(
        plane,
        Instance::new_model(
            na::Matrix4::new_translation(&na::Vector3::new(0.0, 0.0, -2.0))
                * na::Matrix4::new_scaling(1000.0),
        ),
        light_gray,
    );

    scene.add_object_with_material(
        teapot,
        Instance::new_model(
            na::Matrix4::new_translation(&na::Vector3::new(0.0, 0.0, -2.0))
                * na::Matrix4::new_rotation(na::Vector3::y() * 33.0f32.to_radians())
                * na::Matrix4::new_scaling(1.0),
        ),
        lily,
    );

    scene.add_object_with_material(
        teapot,
        Instance::new_model(
            na::Matrix4::new_translation(&na::Vector3::new(-2.0, 0.0, -10.0))
                * na::Matrix4::new_rotation(na::Vector3::y() * 33.0f32.to_radians())
                * na::Matrix4::new_scaling(1.0),
        ),
        lily,
    );

    scene.add_object_with_material(
        teapot,
        Instance::new_model(
            na::Matrix4::new_translation(&na::Vector3::new(-6.0, 0.0, -22.0))
                * na::Matrix4::new_rotation(na::Vector3::y() * 33.0f32.to_radians())
                * na::Matrix4::new_scaling(1.0),
        ),
        lily,
    );

    scene.add_object(
        maya,
        Instance::new_model(na::Matrix4::new_translation(&na::Vector3::new(
            1.0, 0.0, 1.7,
        ))),
    );

    let projection_mat =
        na::Matrix4::new_perspective(gpu.aspect_ratio(), 45.0f32.to_radians(), 0.1, 100.0);

    let projection: GpuProjection = GpuProjection::new(projection_mat, &gpu.device)?;
    let projection_mat = wgpu_projection(projection_mat);

    let mut camera = GpuCamera::new(
        Camera::new(
            na::Point3::new(0.0, 18.0, 14.0),
            -45.0f32.to_radians(),
            270.0f32.to_radians(),
        ),
        &gpu.device,
    )?;

    let mut lights = LightScene::default();

    lights.new_directional(
        na::Vector3::new(-0.5, -0.5, -0.5).normalize(),
        na::Vector3::new(0.1, 0.1, 0.1),
        na::Vector3::new(0.5, 0.5, 0.5),
        na::Vector3::new(0.3, 0.3, 0.3),
    );

    lights.new_spot(
        na::Vector3::new(0.0, 10.0, 0.0),
        na::Vector3::new(0.0, -1.0, 0.0),
        na::Vector3::new(0.1, 0.1, 0.1),
        na::Vector3::new(0.3, 0.2, 0.8),
        na::Vector3::new(0.4, 0.4, 0.4),
        30.0f32.to_radians(),
        na::Vector3::new(1.0, 0.09, 0.032),
    );

    lights.new_point(
        na::Vector3::new(1.0, 0.5, 4.0),
        na::Vector3::new(0.1, 0.1, 0.1),
        na::Vector3::new(0.8, 0.1, 0.1),
        na::Vector3::new(0.8, 0.1, 0.1),
        na::Vector3::new(1.0, 0.09, 0.0032),
    );

    Ok((
        scene,
        material_atlas,
        lights,
        camera,
        projection,
        wgpu_projection(projection_mat),
        HashMap::default(),
    ))
}

pub fn normal_mapping_test(gpu: &Gpu) -> Result<TestScene> {
    let mut scene = Scene::default();
    let mut material_atlas = MaterialAtlas::new(gpu);

    let plane_uv = MeshBuilder::new()
        .with_geometry(Plane::geometry_tan_space())
        .with_texture_uvs(Plane::uvs())
        .build()?;

    let plane = MeshBuilder::new()
        .with_geometry(Plane::geometry())
        .build()?;

    let brickwall_material = material_atlas.add_phong_textured_normal(
        gpu,
        "./textures/brickwall_diffuse.jpg",
        SpecularTexture::FullDiffuse,
        "./textures/brickwall_normal.jpg",
    )?;

    let plane = scene.load_model(SceneModelBuilder::default().with_meshes(vec![plane]));

    let brickwall = scene.load_model(
        SceneModelBuilder::default()
            .with_meshes(vec![plane_uv])
            .with_local_materials(vec![brickwall_material]),
    );

    let yellow = material_atlas.add_phong_solid(
        gpu,
        na::Vector4::new(1.0, 1.0, 0.0, 0.0),
        na::Vector4::new(0.0, 0.0, 0.0, 0.0),
        na::Vector4::new(0.0, 0.0, 0.0, 32.0),
    )?;

    scene.add_object_with_material(
        plane,
        Instance::new_model(
            na::Matrix4::new_translation(&na::Vector3::new(0.5, 1.0, 0.3))
                * na::Matrix4::new_rotation(na::Vector3::x() * 90.0f32.to_radians())
                * na::Matrix4::new_scaling(0.2),
        ),
        yellow,
    );

    let mut lights = LightScene::default();
    lights.new_point(
        na::Vector3::new(0.5, 1.0, 0.3),
        na::Vector3::new(0.1, 0.1, 0.1),
        na::Vector3::new(1.0, 1.0, 1.0),
        na::Vector3::new(0.2, 0.2, 0.2),
        na::Vector3::new(1.0, 0.7, 1.8),
    );

    let wall = scene.add_object(
        brickwall,
        Instance::new_model(na::Matrix4::new_rotation(
            na::Vector3::x() * 90.0f32.to_radians(),
        )),
    );

    let camera = GpuCamera::new(
        Camera::new(
            na::Point3::new(0.0, 0.0, 3.0),
            0.0f32.to_radians(),
            270.0f32.to_radians(),
        ),
        &gpu.device,
    )?;

    let projection_mat =
        na::Matrix4::new_perspective(gpu.aspect_ratio(), 45.0f32.to_radians(), 0.1, 100.0);

    let projection: GpuProjection = GpuProjection::new(projection_mat, &gpu.device)?;

    let mut scene_stuff = HashMap::new();
    scene_stuff.insert("brickwall".to_string(), wall);

    Ok((
        scene,
        material_atlas,
        lights,
        camera,
        projection,
        wgpu_projection(projection_mat),
        scene_stuff,
    ))
}
