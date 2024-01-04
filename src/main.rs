use anyhow::{Context, Result};
use nalgebra as na;

use winit::{
    event::*, event_loop::EventLoop, keyboard::PhysicalKey, window::Window, window::WindowBuilder,
};

use std::{borrow::Cow, path::Path, time::Instant};

// OpenGL Normalized Coordinate System (right-handed):
// x: [-1.0, 1.0]
// y: [-1.0, 1.0]
// z: [-1.0, 1.0]

// WebGPU Normalized Coordinate System (left-handed):
// x: [-1.0, 1.0]
// y: [-1.0, 1.0]
// z: [0.0, 1.0]

// So we're basically scaling z by 0.5 (m33) so it becomes [-0.5, 0.5] and then translate by 0.5 (m34)
#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: na::Matrix4<f32> = na::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

const MOVE_DELTA: f32 = 0.25;
const TILT_DELTA: f32 = 1.0;

// Scale matrix:
// [sx 0 0 0]
// [0 sy 0 0]
// [0 0 sz 0]
// [0 0  0 1]

// Translation matrix:
// [1 0 0 tx]
// [0 1 0 ty]
// [0 0 1 tz]
// [0 0 0 1]

#[derive(Debug)]
struct Model {
    vertices: Vec<na::Vector3<f32>>,
    indices: Vec<u32>,
}

struct Camera {
    position: na::Point3<f32>,
    delta: na::Vector3<f32>,
    pitch: f32,
    yaw: f32,
}

struct Plane {
    normal: na::Vector3<f32>,
}

struct Object {
    model: Model,
    model_mat: na::Matrix4<f32>,
    albedo: na::Vector3<f32>,
}

struct GpuObject {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    model_mat_buf: wgpu::Buffer,
    inv_model_mat_transp_buf: wgpu::Buffer,
    color_buf: wgpu::Buffer,
    num_indices: u64,
    bind_group: Option<wgpu::BindGroup>,
}

struct Cube {
    size: f32,
}

impl Cube {
    fn new(size: f32) -> Self {
        Self { size }
    }

    fn model(&self) -> Model {
        use na::Vector3 as Vec3;

        let mut model = Model {
            vertices: vec![],
            indices: vec![],
        };

        let Model {
            ref mut vertices,
            ref mut indices,
        } = model;

        let half_size = self.size / 2.0;
        vertices.push(Vec3::new(-half_size, half_size, -half_size)); // front-tl
        vertices.push(Vec3::new(half_size, half_size, -half_size)); // front-tr
        vertices.push(Vec3::new(-half_size, -half_size, -half_size)); // front-bl
        vertices.push(Vec3::new(half_size, -half_size, -half_size)); // front-br
        vertices.push(Vec3::new(-half_size, half_size, half_size)); // back-tl
        vertices.push(Vec3::new(half_size, half_size, half_size)); // back-tr
        vertices.push(Vec3::new(-half_size, -half_size, half_size)); // back-bl
        vertices.push(Vec3::new(half_size, -half_size, half_size)); // back-br

        indices.push(0);
        indices.push(2);
        indices.push(1);

        indices.push(1);
        indices.push(2);
        indices.push(3);

        indices.push(4);
        indices.push(6);
        indices.push(5);

        indices.push(5);
        indices.push(6);
        indices.push(7);

        indices.push(0);
        indices.push(2);
        indices.push(4);

        indices.push(4);
        indices.push(6);
        indices.push(2);

        indices.push(1);
        indices.push(3);
        indices.push(5);

        indices.push(5);
        indices.push(7);
        indices.push(3);

        indices.push(0);
        indices.push(4);
        indices.push(1);

        indices.push(1);
        indices.push(5);
        indices.push(4);

        indices.push(2);
        indices.push(6);
        indices.push(3);

        indices.push(3);
        indices.push(7);
        indices.push(6);

        model
    }

    fn normals(&self) -> Vec<na::Vector3<f32>> {
        let center = na::Vector3::zeros();
        self.model()
            .vertices
            .iter()
            .copied()
            .map(|v| (v - center).normalize())
            .collect()
    }
}

struct Light {
    position: na::Point3<f32>,
    color: na::Vector3<f32>,
}

impl GpuObject {
    fn new(object: &Object, device: &wgpu::Device, normals: Vec<na::Vector3<f32>>) -> Result<Self> {
        use wgpu::util::DeviceExt;
        let Object {
            model,
            model_mat,
            albedo,
        } = object;

        let vertex_buf_contents = model
            .vertices
            .iter()
            .copied()
            .zip(normals.iter().copied())
            .flat_map(|(v, n)| [v, n])
            .collect::<Vec<_>>();

        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&vertex_buf_contents),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&model.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let mut buf = encase::UniformBuffer::new(vec![]);
        buf.write(&albedo)?;
        let color_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: buf.into_inner().as_slice(),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let mut buf = encase::UniformBuffer::new(vec![]);
        buf.write(&model_mat)?;
        let model_mat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: buf.into_inner().as_slice(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let mut buf = encase::UniformBuffer::new(vec![]);
        buf.write(&model_mat.try_inverse().unwrap().transpose())?;
        let inv_model_mat_transp_buf =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: buf.into_inner().as_slice(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        Ok(Self {
            vertex_buf,
            index_buf,
            model_mat_buf,
            inv_model_mat_transp_buf,
            num_indices: model.indices.len() as u64,
            bind_group: None,
            color_buf,
        })
    }

    fn update_model_mat(&self, queue: &wgpu::Queue, model_mat: &na::Matrix4<f32>) {
        let mut buf = encase::UniformBuffer::new(vec![]);
        buf.write(&model_mat).unwrap();
        queue.write_buffer(&self.model_mat_buf, 0, buf.into_inner().as_slice());
    }

    fn init_bind_group(
        &mut self,
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        proj_buf: &wgpu::Buffer,
        invproj_buf: &wgpu::Buffer,
        invcamera_buf: &wgpu::Buffer,
        light_mat: &wgpu::Buffer,
        layout: &wgpu::BindGroupLayout,
    ) {
        self.bind_group.get_or_insert_with(|| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: proj_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: invproj_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: invcamera_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.model_mat_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.inv_model_mat_transp_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.color_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: light_mat.as_entire_binding(),
                    },
                ],
            })
        });
    }

    fn bind_group(&self) -> &wgpu::BindGroup {
        self.bind_group.as_ref().unwrap()
    }
}

impl Object {
    fn new(model: Model, model_mat: na::Matrix4<f32>, albedo: na::Vector3<f32>) -> Self {
        Self {
            model,
            model_mat,
            albedo,
        }
    }

    fn as_gpu(&self, device: &wgpu::Device, normals: Vec<na::Vector3<f32>>) -> Result<GpuObject> {
        GpuObject::new(self, device, normals)
    }
}

impl Plane {
    fn new(normal: na::Vector3<f32>) -> Self {
        Self { normal }
    }

    fn model(&self) -> Model {
        let mut model = Model {
            vertices: vec![],
            indices: vec![],
        };

        let z = self.normal.cross(&na::Vector3::y()).normalize();
        let x = self.normal.cross(&z).normalize();

        let tl = -x - z;
        let tr = x - z;
        let bl = -x + z;
        let br = x + z;

        model.vertices.push(tl);
        model.vertices.push(tr);
        model.vertices.push(bl);
        model.vertices.push(br);

        model.indices.push(0);
        model.indices.push(1);
        model.indices.push(2);
        model.indices.push(1);
        model.indices.push(3);
        model.indices.push(2);

        model
    }

    fn normals(&self) -> Vec<na::Vector3<f32>> {
        vec![self.normal; 4]
    }
}

impl Camera {
    fn new(position: na::Point3<f32>, pitch: f32, yaw: f32) -> Self {
        Self {
            position,
            delta: na::Vector3::zeros(),
            pitch,
            yaw,
        }
    }

    fn fly(&mut self, d: f32) {
        self.delta += na::Vector3::y() * d;
    }

    fn strafe(&mut self, d: f32) {
        let target = na::Vector3::new(
            self.pitch.cos() * self.yaw.cos(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.sin(),
        );

        let right = target.cross(&na::Vector3::y()).normalize();
        self.delta += right * d;
    }

    fn forwards(&mut self, d: f32) {
        let target = na::Vector3::new(
            self.pitch.cos() * self.yaw.cos(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.sin(),
        )
        .normalize();

        self.delta += target * d;
    }

    fn tilt_horizontally(&mut self, d: f32) {
        self.yaw += d;
    }

    fn tilt_vertically(&mut self, d: f32) {
        self.pitch += d;
    }

    fn look_at_matrix(&self) -> na::Matrix4<f32> {
        let target = na::Vector3::new(
            self.pitch.cos() * self.yaw.cos(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.sin(),
        );

        let position_now = self.position + self.delta;
        let target_now = position_now + target;

        na::Matrix4::look_at_rh(&position_now, &target_now, &na::Vector3::y())
    }
}

fn read_obj(path: impl AsRef<Path>) -> Result<Model> {
    use std::fs::File;
    use std::io::{prelude::*, BufReader};

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let reader = BufReader::new(File::open(path)?);

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
                vertices.push(na::Vector3::new(x, y, z));
            }
            "f" => {
                let mut iter = line.split_whitespace();
                iter.next();
                let x = iter.next().unwrap().parse::<u32>().unwrap();
                let y = iter.next().unwrap().parse::<u32>().unwrap();
                let z = iter.next().unwrap().parse::<u32>().unwrap();
                indices.push(x - 1);
                indices.push(y - 1);
                indices.push(z - 1);
            }
            _ => {}
        }
    }

    Ok(Model { vertices, indices })
}

fn calculate_normals(model: &Model) -> Vec<na::Vector3<f32>> {
    let mut normals = vec![na::Vector3::zeros(); model.vertices.len()];

    for i in (0..model.indices.len()).step_by(3) {
        let i0 = model.indices[i] as usize;
        let i1 = model.indices[i + 1] as usize;
        let i2 = model.indices[i + 2] as usize;

        let v0 = model.vertices[i0];
        let v1 = model.vertices[i1];
        let v2 = model.vertices[i2];

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

    normals
}

async fn run(event_loop: EventLoop<()>, window: Window) -> Result<()> {
    let mut size = window.inner_size();
    size.width = size.width.max(1);
    size.height = size.height.max(1);

    let aspect_ratio = size.width as f32 / size.height as f32;

    let instance = wgpu::Instance::default();
    let surface = unsafe { instance.create_surface(&window).unwrap() };

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            // Request an adapter which can render to our surface
            compatible_surface: Some(&surface),
        })
        .await
        .ok_or(anyhow::anyhow!("Failed to find an appropriate adapter"))?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
                limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
            },
            None,
        )
        .await
        .context("Failed to create device")?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/phong.wgsl"))),
    });

    let mut objects = vec![];
    {
        let mut model_mat = na::Matrix4::identity();
        model_mat *= na::Matrix4::new_translation(&na::Vector3::new(0.0, 0.0, 0.0));
        model_mat *= na::Matrix4::new_rotation(na::Vector3::x() * 270.0f32.to_radians());
        model_mat *= na::Matrix4::new_scaling(100.0);

        let plane = Plane::new(na::Vector3::z());
        let normals = plane.normals();

        let plane_obj = Object::new(plane.model(), model_mat, na::Vector3::new(0.6, 0.6, 0.6));
        let plane_gpu = plane_obj.as_gpu(&device, normals)?;
        objects.push(plane_gpu);
    }

    {
        let mut model_mat = na::Matrix4::identity();
        model_mat *= na::Matrix4::new_translation(&na::Vector3::new(6.0, 2.001, 0.0));
        model_mat *= na::Matrix4::new_rotation(na::Vector3::y() * 45.0f32.to_radians());
        model_mat *= na::Matrix4::new_scaling(1.0);

        let cube = Cube::new(4.0);
        let normals = cube.normals();

        let cube_obj = Object::new(cube.model(), model_mat, na::Vector3::new(0.8, 0.2, 0.2));
        let cube_gpu = cube_obj.as_gpu(&device, normals)?;
        objects.push(cube_gpu);
    }

    {
        let teapot = read_obj("./models/teapot.obj")?;
        let normals = calculate_normals(&teapot);
        let model_mat = OPENGL_TO_WGPU_MATRIX
            * na::Matrix4::new_translation(&na::Vector3::new(-5.0, 0.0, 0.0))
            * na::Matrix4::new_rotation(na::Vector3::y() * 33.0f32.to_radians())
            * na::Matrix4::new_scaling(3.0);

        let teapot_obj = Object::new(teapot, model_mat, na::Vector3::new(0.5, 0.5, 1.0));
        let teapot_gpu = teapot_obj.as_gpu(&device, normals)?;
        objects.push(teapot_gpu);
    }

    {
        let light_model = Cube::new(0.5);
        let light_normals = light_model.normals();
        let light_model_mat = na::Matrix4::<f32>::identity()
            * na::Matrix4::new_translation(&na::Vector3::new(20.0, 2.0, 0.0))
            * na::Matrix4::new_scaling(1.0);

        let light_obj = Object::new(
            light_model.model(),
            light_model_mat,
            na::Vector3::new(1.0, 1.0, 1.0),
        );

        let light_gpu = light_obj.as_gpu(&device, light_normals)?;
        objects.push(light_gpu);
    }

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0];

    use wgpu::util::DeviceExt;

    // Projection matrices are flipping the Z coordinate in nalgebra, see: https://nalgebra.org/docs/user_guide/projections/
    let projection = OPENGL_TO_WGPU_MATRIX
        * na::Matrix4::new_perspective(aspect_ratio, 45.0f32.to_radians(), 0.1, 100.0);

    let proj_buf: wgpu::Buffer;
    {
        let mut buf = encase::UniformBuffer::new(vec![]);
        buf.write(&(OPENGL_TO_WGPU_MATRIX * projection))?;
        proj_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: buf.into_inner().as_slice(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
    }

    let invproj_buf: wgpu::Buffer;
    {
        let mut buf = encase::UniformBuffer::new(vec![]);
        buf.write(&(OPENGL_TO_WGPU_MATRIX * projection).try_inverse().unwrap())?;
        invproj_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: buf.into_inner().as_slice(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
    }

    let mut camera: Camera = Camera::new(
        na::Point3::new(0.0, 4.0, -40.0),
        0.0f32.to_radians(),
        90.0f32.to_radians(),
    );

    let invcamera_buf: wgpu::Buffer;
    {
        let mut buf = encase::UniformBuffer::new(vec![]);
        buf.write(&camera.look_at_matrix().try_inverse().unwrap())?;
        invcamera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: buf.into_inner().as_slice(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
    }

    let camera_buf: wgpu::Buffer;
    {
        let mut buf = encase::UniformBuffer::new(vec![]);
        buf.write(&camera.look_at_matrix())?;
        camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: buf.into_inner().as_slice(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
    }

    let obj_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&obj_bgl],
        push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: (std::mem::size_of::<na::Vector3<f32>>() * 2) as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        format: wgpu::VertexFormat::Float32x3,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        offset: std::mem::size_of::<na::Vector3<f32>>() as wgpu::BufferAddress,
                        format: wgpu::VertexFormat::Float32x3,
                        shader_location: 1,
                    },
                ],
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(swapchain_format.into())],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: swapchain_format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: swapchain_capabilities.alpha_modes[0],
        view_formats: vec![],
    };

    surface.configure(&device, &config);

    let mut depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let mut light_model_mat = na::Matrix4::<f32>::identity()
        * na::Matrix4::new_translation(&na::Vector3::new(10.0, 10.0, 0.0))
        * na::Matrix4::new_scaling(1.0);

    let window = &window;
    let camera = &mut camera;
    let light_model_mat = &mut light_model_mat;

    let mut light_model_mat_contents = encase::UniformBuffer::new(vec![]);
    light_model_mat_contents.write(&light_model_mat)?;

    let light_model_mat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: light_model_mat_contents.into_inner().as_slice(),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let mut delta = Instant::now();
    let delta = &mut delta;

    event_loop
        .run(move |event, target| {
            // Have the closure take ownership of the resources.
            // `event_loop.run` never returns, therefore we must do this to ensure
            // the resources are properly cleaned up.
            // let _ = (&instance, &adapter, &shader, &pipeline_layout);

            use winit::keyboard::KeyCode;
            if let Event::WindowEvent {
                window_id: _,
                event,
            } = event
            {
                match event {
                    WindowEvent::Resized(new_size) => {
                        // Reconfigure the surface with the new size
                        config.width = new_size.width.max(1);
                        config.height = new_size.height.max(1);
                        surface.configure(&device, &config);
                        depth_tex = device.create_texture(&wgpu::TextureDescriptor {
                            label: None,
                            size: wgpu::Extent3d {
                                width: new_size.width,
                                height: new_size.height,
                                depth_or_array_layers: 1,
                            },
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: wgpu::TextureFormat::Depth32Float,
                            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                            view_formats: &[],
                        });
                        // On macos the window needs to be redrawn manually after resizing
                        window.request_redraw();
                    }
                    WindowEvent::CloseRequested => {
                        target.exit();
                    }
                    WindowEvent::RedrawRequested => {
                        let delta_t = delta.elapsed().as_secs_f32();

                        let light_rotation =
                            (((delta_t / (1.0 / 60.0)) as u64 % 360) as f32).to_radians();

                        light_model_mat.column_mut(3).x = light_rotation.cos() * 18.0;
                        light_model_mat.column_mut(3).z = light_rotation.sin() * 18.0;

                        let frame = surface.get_current_texture().unwrap();
                        let view = frame
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());

                        {
                            let mut buf = encase::UniformBuffer::new(vec![]);
                            buf.write(&camera.look_at_matrix()).unwrap();
                            queue.write_buffer(&camera_buf, 0, buf.into_inner().as_slice());
                        }

                        {
                            let mut buf = encase::UniformBuffer::new(vec![]);
                            buf.write(&camera.look_at_matrix().try_inverse().unwrap())
                                .unwrap();
                            queue.write_buffer(&invcamera_buf, 0, buf.into_inner().as_slice());
                        }

                        {
                            let mut light_model_mat_contents = encase::UniformBuffer::new(vec![]);
                            light_model_mat_contents.write(light_model_mat).unwrap();
                            queue.write_buffer(
                                &light_model_mat_buf,
                                0,
                                light_model_mat_contents.into_inner().as_slice(),
                            );
                            objects[objects.len() - 1].update_model_mat(&queue, light_model_mat);
                        }

                        for (idx, object) in objects.iter_mut().enumerate() {
                            object.init_bind_group(
                                &device,
                                &camera_buf,
                                &proj_buf,
                                &invproj_buf,
                                &invcamera_buf,
                                &light_model_mat_buf,
                                &obj_bgl,
                            );

                            let mut encoder = device
                                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                            {
                                let obj_bg = object.bind_group();
                                let depth_view =
                                    depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
                                let mut rpass =
                                    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                        label: None,
                                        color_attachments: &[Some(
                                            wgpu::RenderPassColorAttachment {
                                                view: &view,
                                                resolve_target: None,
                                                ops: wgpu::Operations {
                                                    load: wgpu::LoadOp::Load,
                                                    store: wgpu::StoreOp::Store,
                                                },
                                            },
                                        )],
                                        depth_stencil_attachment: Some(
                                            wgpu::RenderPassDepthStencilAttachment {
                                                view: &depth_view,
                                                depth_ops: Some(wgpu::Operations {
                                                    load: if idx == 0 {
                                                        wgpu::LoadOp::Clear(1.0)
                                                    } else {
                                                        wgpu::LoadOp::Load
                                                    },
                                                    store: wgpu::StoreOp::Store,
                                                }),
                                                stencil_ops: None,
                                            },
                                        ),
                                        timestamp_writes: None,
                                        occlusion_query_set: None,
                                    });
                                rpass.set_pipeline(&render_pipeline);
                                rpass.set_vertex_buffer(0, object.vertex_buf.slice(..));
                                rpass.set_index_buffer(
                                    object.index_buf.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                rpass.set_bind_group(0, obj_bg, &[]);
                                rpass.draw_indexed(0..object.num_indices as u32, 0, 0..1);
                            }
                            queue.submit(Some(encoder.finish()));
                        }

                        frame.present();
                        window.request_redraw();
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        if event.state.is_pressed() {
                            match event.physical_key {
                                PhysicalKey::Code(KeyCode::KeyA) => {
                                    camera.strafe(-MOVE_DELTA);
                                }
                                PhysicalKey::Code(KeyCode::KeyD) => {
                                    camera.strafe(MOVE_DELTA);
                                }
                                PhysicalKey::Code(KeyCode::KeyQ) => {
                                    camera.fly(MOVE_DELTA);
                                }
                                PhysicalKey::Code(KeyCode::KeyZ) => {
                                    camera.fly(-MOVE_DELTA);
                                }
                                PhysicalKey::Code(KeyCode::KeyW) => {
                                    camera.forwards(MOVE_DELTA);
                                }
                                PhysicalKey::Code(KeyCode::KeyS) => camera.forwards(-MOVE_DELTA),
                                PhysicalKey::Code(KeyCode::ArrowLeft) => {
                                    camera.tilt_horizontally(-TILT_DELTA.to_radians());
                                }
                                PhysicalKey::Code(KeyCode::ArrowRight) => {
                                    camera.tilt_horizontally(TILT_DELTA.to_radians());
                                }
                                PhysicalKey::Code(KeyCode::ArrowUp) => {
                                    camera.tilt_vertically(TILT_DELTA.to_radians());
                                }
                                PhysicalKey::Code(KeyCode::ArrowDown) => {
                                    camera.tilt_vertically(-TILT_DELTA.to_radians());
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                };
            }
        })
        .unwrap();

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new().build(&event_loop)?;

    run(event_loop, window).await?;

    Ok(())
}
