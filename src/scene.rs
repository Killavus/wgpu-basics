use anyhow::Result;
use nalgebra as na;

type FMat4x4 = na::Matrix4<f32>;

use crate::{
    gpu::Gpu,
    material::MaterialId,
    mesh::{
        Mesh, MeshVertexArrayType, PNTBUV_SLOTS, PNTBUV_STRIDE, PNUV_SLOTS, PNUV_STRIDE, PN_SLOTS,
        PN_STRIDE,
    },
};

const MAX_INSTANCE_BUFFER_GROWTH: usize = 128;

struct ModelDescriptor {
    mesh_r: (usize, usize),
    local_material_r: Option<(usize, usize)>,
    local_instances_r: Option<(usize, usize)>,
}

pub const MODEL_INSTANCE_STRIDE: usize = std::mem::size_of::<FMat4x4>() * 2;

#[derive(Clone, Copy, Debug)]
pub enum InstanceArrayType {
    // Model = Mat4x4 model matrix + Mat4x4 inverse transpose model matrix
    Model,
}

impl InstanceArrayType {
    pub fn stride(&self) -> usize {
        match self {
            Self::Model => MODEL_INSTANCE_STRIDE,
        }
    }
}

#[derive(Default)]
pub struct SceneStorage {
    meshes: Vec<Mesh>,
    instances: Vec<Instance>,
    local_materials: Vec<MaterialId>,
    model_descriptors: Vec<ModelDescriptor>,
}

#[derive(Default)]
pub struct Scene {
    storage: SceneStorage,
    objects: Vec<SceneObject>,
}

#[derive(Clone, Copy)]
pub struct Instance {
    model: FMat4x4,
    model_invt: FMat4x4,
    spec: InstanceSpec,
}

#[derive(Clone, Copy)]
pub enum InstanceSpec {
    None,
}

impl Instance {
    const PN_MODEL_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: MODEL_INSTANCE_STRIDE as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &wgpu::vertex_attr_array![
            PN_SLOTS => Float32x4,
            PN_SLOTS + 1 => Float32x4,
            PN_SLOTS + 2 => Float32x4,
            PN_SLOTS + 3 => Float32x4,
            PN_SLOTS + 4 => Float32x4,
            PN_SLOTS + 5 => Float32x4,
            PN_SLOTS + 6 => Float32x4,
            PN_SLOTS + 7 => Float32x4,
        ],
    };

    const PNUV_MODEL_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: MODEL_INSTANCE_STRIDE as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &wgpu::vertex_attr_array![
            PNUV_SLOTS => Float32x4,
            PNUV_SLOTS + 1 => Float32x4,
            PNUV_SLOTS + 2 => Float32x4,
            PNUV_SLOTS + 3 => Float32x4,
            PNUV_SLOTS + 4 => Float32x4,
            PNUV_SLOTS + 5 => Float32x4,
            PNUV_SLOTS + 6 => Float32x4,
            PNUV_SLOTS + 7 => Float32x4,
        ],
    };

    const PNTBUV_MODEL_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: MODEL_INSTANCE_STRIDE as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &wgpu::vertex_attr_array![
            PNTBUV_SLOTS => Float32x4,
            PNTBUV_SLOTS + 1 => Float32x4,
            PNTBUV_SLOTS + 2 => Float32x4,
            PNTBUV_SLOTS + 3 => Float32x4,
            PNTBUV_SLOTS + 4 => Float32x4,
            PNTBUV_SLOTS + 5 => Float32x4,
            PNTBUV_SLOTS + 6 => Float32x4,
            PNTBUV_SLOTS + 7 => Float32x4,
        ],
    };

    pub fn new_model(model: FMat4x4) -> Self {
        Self {
            model,
            model_invt: model.try_inverse().unwrap().transpose(),
            spec: InstanceSpec::None,
        }
    }

    pub fn update_from_object(self, object_instance: &Instance) -> Self {
        Self::new_model(object_instance.model * self.model)
    }

    pub fn copy_to(&self, target: &mut Vec<u8>) {
        target.extend(bytemuck::cast_slice(&[self.model, self.model_invt]));
    }

    pub fn pn_model_instance_layout() -> wgpu::VertexBufferLayout<'static> {
        Self::PN_MODEL_LAYOUT
    }

    pub fn pnuv_model_instance_layout() -> wgpu::VertexBufferLayout<'static> {
        Self::PNUV_MODEL_LAYOUT
    }

    pub fn pntbuv_model_instance_layout() -> wgpu::VertexBufferLayout<'static> {
        Self::PNTBUV_MODEL_LAYOUT
    }
}

impl Scene {
    pub fn load_model(&mut self, model_builder: SceneModelBuilder) -> SceneModel {
        self.storage.load_model(model_builder)
    }

    pub fn add_object(&mut self, model: SceneModel, instance: Instance) -> SceneObjectId {
        let instance_idx = self.storage.instances.len();
        self.storage.instances.push(instance);

        let mesh_transforms_r = self.append_mesh_transforms(model, instance);

        let object = SceneObject {
            instance_idx,
            material_idx: None,
            mesh_instances_r: mesh_transforms_r,
            model_idx: model.0,
        };

        let object_idx = self.objects.len();
        self.objects.push(object);

        SceneObjectId(object_idx)
    }

    fn append_mesh_transforms(&mut self, model: SceneModel, instance: Instance) -> (usize, usize) {
        let mesh_count = self.storage.model_descriptors[model.0].mesh_r.1
            - self.storage.model_descriptors[model.0].mesh_r.0;

        let mesh_transforms_r = (
            self.storage.instances.len(),
            self.storage.instances.len() + mesh_count,
        );

        let mut remaining_instances = mesh_count;
        if let Some((local_instance_s, local_instance_e)) =
            self.storage.model_descriptors[model.0].local_instances_r
        {
            let local_transforms = self.storage.instances[local_instance_s..local_instance_e]
                .iter()
                .map(|local_instance| local_instance.update_from_object(&instance))
                .collect::<Vec<_>>();

            self.storage.instances.extend(local_transforms);
            remaining_instances = mesh_count - (local_instance_e - local_instance_s);
        }

        if remaining_instances > 0 {
            self.storage
                .instances
                .extend(std::iter::repeat(instance).take(remaining_instances));
        }

        mesh_transforms_r
    }

    pub fn add_object_with_material(
        &mut self,
        model: SceneModel,
        instance: Instance,
        material: MaterialId,
    ) -> SceneObjectId {
        let transform_idx = self.storage.instances.len();
        self.storage.instances.push(instance);

        let mesh_transforms_r = self.append_mesh_transforms(model, instance);

        let object = SceneObject {
            instance_idx: transform_idx,
            material_idx: Some(material),
            mesh_instances_r: mesh_transforms_r,
            model_idx: model.0,
        };

        let object_idx: usize = self.objects.len();
        self.objects.push(object);

        SceneObjectId(object_idx)
    }
}

#[derive(Debug)]
struct SceneObject {
    instance_idx: usize,
    material_idx: Option<MaterialId>,
    mesh_instances_r: (usize, usize),
    model_idx: usize,
}

pub struct SceneObjectId(usize);

#[derive(Default)]
pub struct SceneModelBuilder {
    meshes: Vec<Mesh>,
    local_instances: Option<Vec<Instance>>,
    local_materials: Option<Vec<MaterialId>>,
}

impl SceneModelBuilder {
    pub fn with_meshes(mut self, meshes: Vec<Mesh>) -> Self {
        self.meshes = meshes;
        self
    }

    pub fn with_local_instances(mut self, instances: Vec<Instance>) -> Self {
        self.local_instances = Some(instances);
        self
    }

    pub fn with_local_materials(mut self, materials: Vec<MaterialId>) -> Self {
        self.local_materials = Some(materials);
        self
    }
}

#[derive(Clone, Copy)]
pub struct SceneModel(usize);

impl SceneStorage {
    fn load_model(&mut self, builder: SceneModelBuilder) -> SceneModel {
        let mesh_r = (self.meshes.len(), self.meshes.len() + builder.meshes.len());
        for mesh in builder.meshes {
            self.meshes.push(mesh);
        }

        let mut local_transform_r = None;
        if let Some(instances) = builder.local_instances {
            local_transform_r =
                Some((self.instances.len(), self.instances.len() + instances.len()));

            for instance in instances {
                self.instances.push(instance);
            }
        }

        let mut local_material_r = None;
        if let Some(materials) = builder.local_materials {
            local_material_r = Some((
                self.local_materials.len(),
                self.local_materials.len() + materials.len(),
            ));

            for material in materials {
                self.local_materials.push(material);
            }
        }

        let model_idx = self.model_descriptors.len();
        self.model_descriptors.push(ModelDescriptor {
            mesh_r,
            local_material_r,
            local_instances_r: local_transform_r,
        });

        SceneModel(model_idx)
    }
}

struct VertexBuffers {
    pntbuv_buffer: Option<wgpu::Buffer>,
    pnuv_buffer: Option<wgpu::Buffer>,
    pn_buffer: Option<wgpu::Buffer>,
}

// This representation works assuming that Features::FIRST_INSTANCE is present on the device.
struct InstanceBuffers {
    model_ib: Option<wgpu::Buffer>,
}

pub struct GpuScene {
    instances: Vec<Instance>,
    materials: Vec<MaterialId>,
    vertex_buffers: VertexBuffers,
    instance_buffers: InstanceBuffers,
    index_buffer: wgpu::Buffer,
    draw_buffers: DrawBuffers,
    mesh_descriptors: Vec<MeshDescriptor>,
    draw_calls: Vec<DrawCall>,
}

#[derive(Debug)]
pub struct DrawCall {
    pub indexed: bool,
    pub draw_buffer_offset: wgpu::BufferAddress,
    pub material_id: MaterialId,
    pub vertex_array_type: MeshVertexArrayType,
    pub instance_type: InstanceArrayType,
}

struct DrawBuffers {
    indexed_buffer: Option<wgpu::Buffer>,
    indexed_buffer_count: usize,
    non_indexed_buffer: Option<wgpu::Buffer>,
    non_indexed_buffer_count: usize,
}

struct MeshDescriptor {
    vertex_array_type: MeshVertexArrayType,
    mesh_bank_vertex_no: usize,
    num_vertices: usize,
    index_buffer_index_no: Option<usize>,
    num_indices: Option<usize>,
}

impl GpuScene {
    pub fn new(gpu: &Gpu, scene: Scene) -> Result<Self> {
        let mut index_buffer_contents = vec![];
        let mut mesh_descriptors = Vec::with_capacity(scene.storage.meshes.len());

        let mut pnuv_vertices = vec![];
        let mut pn_vertices = vec![];
        let mut pntbuv_vertices = vec![];

        for mesh in scene.storage.meshes.iter() {
            let mesh_bank = match mesh.vertex_array_type() {
                MeshVertexArrayType::PN => &mut pn_vertices,
                MeshVertexArrayType::PNUV => &mut pnuv_vertices,
                MeshVertexArrayType::PNTBUV => &mut pntbuv_vertices,
            };

            let vertex_stride = match mesh.vertex_array_type() {
                MeshVertexArrayType::PN => PN_STRIDE,
                MeshVertexArrayType::PNUV => PNUV_STRIDE,
                MeshVertexArrayType::PNTBUV => PNTBUV_STRIDE,
            };

            let mesh_bank_offset = mesh_bank.len();
            let num_vertices = mesh.num_vertices();

            mesh.copy_to_mesh_bank(mesh_bank);

            let num_indices = mesh.num_indices();
            let mut index_buffer_offset = None;
            if mesh.is_indexed() {
                index_buffer_offset = Some(index_buffer_contents.len());
                mesh.copy_to_index_buffer(&mut index_buffer_contents);
            }

            mesh_descriptors.push(MeshDescriptor {
                vertex_array_type: mesh.vertex_array_type(),
                mesh_bank_vertex_no: mesh_bank_offset / vertex_stride,
                num_vertices,
                index_buffer_index_no: index_buffer_offset,
                num_indices,
            });
        }

        let index_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("IndexBuffer"),
                contents: bytemuck::cast_slice(&index_buffer_contents),
                usage: wgpu::BufferUsages::INDEX,
            });

        let mut pnuv_buffer = None;
        let mut pn_buffer = None;
        let mut pntbuv_buffer = None;

        use wgpu::util::DeviceExt;
        if !pnuv_vertices.is_empty() {
            pnuv_buffer = Some(
                gpu.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("PNUV Vertex Buffer"),
                        contents: bytemuck::cast_slice(&pnuv_vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    }),
            );
        }

        if !pn_vertices.is_empty() {
            pn_buffer = Some(
                gpu.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("PN Vertex Buffer"),
                        contents: bytemuck::cast_slice(&pn_vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    }),
            );
        }

        if !pntbuv_vertices.is_empty() {
            pntbuv_buffer = Some(gpu.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("PNTBUV Vertex Buffer"),
                    contents: bytemuck::cast_slice(&pntbuv_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                },
            ));
        }

        let vertex_buffers = VertexBuffers {
            pntbuv_buffer,
            pnuv_buffer,
            pn_buffer,
        };

        /* IDEA: Let's keep the same mesh/bind group combos together so we can maximize instancing.
          Instance buffer needs to grow (we potentially want to conditionally add / remove objects dynamically)
          so we allocate MAX_INSTANCE_BUFFER_GROWTH more.
          The same with draw buffers - newly added objects won't benefit from instancing.
        */
        /* REIMPL: This is potentially counter-productive if frustum/occlusion culling gets introduced.
           Reconstruction of all draw buffers will be needed every frame.
           Also keeping track of SceneObjectId <-> InstanceBuffer ranges is going to be required then, but YAGNI.
        */
        use std::collections::BTreeMap;
        let mut instance_banks: BTreeMap<(usize, MaterialId), Vec<u8>> = BTreeMap::new();

        for scene_object in scene.objects {
            let descriptor = &scene.storage.model_descriptors[scene_object.model_idx];

            let mesh_r = descriptor.mesh_r.0..descriptor.mesh_r.1;
            let mut material_r = descriptor
                .local_material_r
                .map(|(s, e)| s..e)
                .unwrap_or(0..0);

            for mesh_idx in mesh_r {
                let material_idx = material_r
                    .next()
                    .map(|idx| scene.storage.local_materials[idx])
                    .or(scene_object.material_idx)
                    .ok_or_else(|| anyhow::anyhow!("No material found for mesh"))?;

                let instance_bank = instance_banks.entry((mesh_idx, material_idx)).or_default();

                let instances_r = scene_object.mesh_instances_r.0..scene_object.mesh_instances_r.1;
                for instance in &scene.storage.instances[instances_r] {
                    instance.copy_to(instance_bank);
                }
            }
        }

        let draw_buffers_count = instance_banks.keys().len();
        let mut instance_buffer_draws = Vec::with_capacity(draw_buffers_count);
        let mut transform_ib_contents: Vec<u8> =
            Vec::with_capacity(instance_banks.values().map(Vec::len).sum());

        for ((mesh_idx, material_id), instance_bank) in instance_banks.into_iter() {
            let instance_bank_offset = transform_ib_contents.len();
            instance_buffer_draws.push((
                instance_bank_offset / MODEL_INSTANCE_STRIDE,
                instance_bank.len() / MODEL_INSTANCE_STRIDE,
                &mesh_descriptors[mesh_idx],
                material_id,
            ));
            transform_ib_contents.extend(instance_bank);
        }

        let mut transform_ib = None;

        if !transform_ib_contents.is_empty() {
            let ib = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("InstanceBuffer:Transform"),
                size: (transform_ib_contents.len()
                    + MAX_INSTANCE_BUFFER_GROWTH * MODEL_INSTANCE_STRIDE)
                    as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            gpu.queue
                .write_buffer(&ib, 0, transform_ib_contents.as_slice());

            transform_ib = Some(ib);
        }

        let instance_buffers = InstanceBuffers {
            model_ib: transform_ib,
        };

        // Now let's create draw buffers...
        let mut indexed_draw_buffer_contents: Vec<u8> = vec![];
        let mut non_indexed_draw_buffer_contents: Vec<u8> = vec![];
        let mut draw_calls = Vec::with_capacity(draw_buffers_count);

        for (ib_first, ib_count, mesh_descriptor, material_id) in instance_buffer_draws {
            let call = DrawCall {
                indexed: mesh_descriptor.index_buffer_index_no.is_some(),
                draw_buffer_offset: if mesh_descriptor.index_buffer_index_no.is_some() {
                    indexed_draw_buffer_contents.len()
                } else {
                    non_indexed_draw_buffer_contents.len()
                } as wgpu::BufferAddress,
                material_id,
                vertex_array_type: mesh_descriptor.vertex_array_type,
                instance_type: InstanceArrayType::Model,
            };

            if call.indexed {
                let args = wgpu::util::DrawIndexedIndirectArgs {
                    index_count: mesh_descriptor.num_indices.unwrap() as u32,
                    instance_count: ib_count as u32,
                    first_index: mesh_descriptor.index_buffer_index_no.unwrap() as u32,
                    base_vertex: mesh_descriptor.mesh_bank_vertex_no as i32,
                    first_instance: ib_first as u32,
                };

                indexed_draw_buffer_contents.extend(bytemuck::cast_slice(&[
                    args.index_count,
                    args.instance_count,
                    args.first_index,
                ]));
                indexed_draw_buffer_contents.extend(bytemuck::cast_slice(&[args.base_vertex]));
                indexed_draw_buffer_contents.extend(bytemuck::cast_slice(&[args.first_instance]));
            } else {
                let args = wgpu::util::DrawIndirectArgs {
                    vertex_count: mesh_descriptor.num_vertices as u32,
                    instance_count: ib_count as u32,
                    first_vertex: mesh_descriptor.mesh_bank_vertex_no as u32,
                    first_instance: ib_first as u32,
                };

                non_indexed_draw_buffer_contents.extend(bytemuck::cast_slice(&[
                    args.vertex_count,
                    args.instance_count,
                    args.first_vertex,
                    args.first_instance,
                ]));
            }

            draw_calls.push(call);
        }

        let indexed_draw_buffer_stride =
            std::mem::size_of::<u32>() * 4 + std::mem::size_of::<i32>();
        let non_indexed_draw_buffer_stride = std::mem::size_of::<u32>() * 4;

        let mut indexed_draw_buffer = None;
        if !indexed_draw_buffer_contents.is_empty() {
            let db = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("DrawBuffer:Indexed"),
                size: (indexed_draw_buffer_contents.len()
                    + indexed_draw_buffer_stride * MAX_INSTANCE_BUFFER_GROWTH)
                    as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            gpu.queue
                .write_buffer(&db, 0, indexed_draw_buffer_contents.as_slice());

            indexed_draw_buffer = Some(db);
        }

        let mut non_indexed_draw_buffer = None;
        if !non_indexed_draw_buffer_contents.is_empty() {
            let db = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("DrawBuffer:NonIndexed"),
                size: (non_indexed_draw_buffer_contents.len()
                    + non_indexed_draw_buffer_stride * MAX_INSTANCE_BUFFER_GROWTH)
                    as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            gpu.queue
                .write_buffer(&db, 0, non_indexed_draw_buffer_contents.as_slice());
            non_indexed_draw_buffer = Some(db);
        }

        let draw_buffers = DrawBuffers {
            indexed_buffer: indexed_draw_buffer,
            indexed_buffer_count: indexed_draw_buffer_contents.len() / indexed_draw_buffer_stride,
            non_indexed_buffer: non_indexed_draw_buffer,
            non_indexed_buffer_count: non_indexed_draw_buffer_contents.len()
                / non_indexed_draw_buffer_stride,
        };

        Ok(Self {
            instances: scene.storage.instances,
            materials: scene.storage.local_materials,
            vertex_buffers,
            instance_buffers,
            index_buffer,
            draw_buffers,
            mesh_descriptors,
            draw_calls,
        })
    }

    pub fn instance_buffer_by_type(&self, instance_type: InstanceArrayType) -> &wgpu::Buffer {
        match instance_type {
            InstanceArrayType::Model => self.instance_buffers.model_ib.as_ref().unwrap(),
        }
    }

    pub fn vertex_buffer_by_type(&self, vertex_type: MeshVertexArrayType) -> &wgpu::Buffer {
        match vertex_type {
            MeshVertexArrayType::PN => self.vertex_buffers.pn_buffer.as_ref().unwrap(),
            MeshVertexArrayType::PNUV => self.vertex_buffers.pnuv_buffer.as_ref().unwrap(),
            MeshVertexArrayType::PNTBUV => self.vertex_buffers.pntbuv_buffer.as_ref().unwrap(),
        }
    }

    pub fn update_instance<F>(&mut self, scene_object_id: SceneObjectId, updater: F)
    where
        F: Fn(&mut Instance) -> Instance,
    {
    }

    pub fn index_buffer(&self) -> &wgpu::Buffer {
        &self.index_buffer
    }

    pub fn draw_calls(&self) -> &[DrawCall] {
        &self.draw_calls
    }

    pub fn indexed_draw_buffer(&self) -> &wgpu::Buffer {
        self.draw_buffers.indexed_buffer.as_ref().unwrap()
    }

    pub fn non_indexed_draw_buffer(&self) -> &wgpu::Buffer {
        self.draw_buffers.non_indexed_buffer.as_ref().unwrap()
    }
}
