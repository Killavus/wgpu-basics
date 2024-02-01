// use na::{Matrix4, Vector3};
// use nalgebra as na;

// use crate::shapes::{GpuModel, Model};

// pub struct WorldModel {
//     model: Model,
//     model_matrices: Vec<Matrix4<f32>>,
//     model_albedo: Vec<Vector3<f32>>,
// }

// impl WorldModel {
//     pub fn new(model: Model) -> Self {
//         Self {
//             model,
//             model_matrices: Vec::new(),
//             model_albedo: Vec::new(),
//         }
//     }

//     pub fn add(&mut self, matrix: Matrix4<f32>, albedo: Vector3<f32>) -> usize {
//         self.model_matrices.push(matrix);
//         self.model_albedo.push(albedo);

//         self.model_matrices.len() - 1
//     }

//     pub fn into_gpu(self, device: &wgpu::Device) -> GpuWorldModel {
//         GpuWorldModel::new(device, self)
//     }
// }

// pub struct GpuWorldModel {
//     gpu_model: GpuModel,
//     model_matrices: Vec<Matrix4<f32>>,
//     model_albedo: Vec<Vector3<f32>>,
//     instance_buf: wgpu::Buffer,
// }

// impl GpuWorldModel {
//     const INSTANCE_LAYOUT: [wgpu::VertexAttribute; 9] = wgpu::vertex_attr_array![
//         GpuModel::VERTEX_ATTRS_MAX_SLOT => Float32x4,
//         GpuModel::VERTEX_ATTRS_MAX_SLOT + 1 => Float32x4,
//         GpuModel::VERTEX_ATTRS_MAX_SLOT + 2 => Float32x4,
//         GpuModel::VERTEX_ATTRS_MAX_SLOT + 3 => Float32x4,
//         GpuModel::VERTEX_ATTRS_MAX_SLOT + 4 => Float32x4,
//         GpuModel::VERTEX_ATTRS_MAX_SLOT + 5 => Float32x4,
//         GpuModel::VERTEX_ATTRS_MAX_SLOT + 6 => Float32x4,
//         GpuModel::VERTEX_ATTRS_MAX_SLOT + 7 => Float32x4,
//         GpuModel::VERTEX_ATTRS_MAX_SLOT + 8 => Float32x3,
//     ];

//     pub fn instance_layout() -> wgpu::VertexBufferLayout<'static> {
//         wgpu::VertexBufferLayout {
//             array_stride: (std::mem::size_of::<Matrix4<f32>>() * 2
//                 + std::mem::size_of::<Vector3<f32>>())
//                 as wgpu::BufferAddress,
//             step_mode: wgpu::VertexStepMode::Instance,
//             attributes: &Self::INSTANCE_LAYOUT,
//         }
//     }

//     pub fn construct_instance_buf(
//         device: &wgpu::Device,
//         model_matrices: &[Matrix4<f32>],
//         model_albedo: &[Vector3<f32>],
//     ) -> wgpu::Buffer {
//         use wgpu::util::DeviceExt;

//         let modelinvt_mats = model_matrices
//             .iter()
//             .copied()
//             .map(|m| m.try_inverse().unwrap().transpose())
//             .collect::<Vec<_>>();

//         let mut instance_data: Vec<u8> = Vec::with_capacity(
//             model_matrices.len() * 2 * std::mem::size_of::<Matrix4<f32>>()
//                 + std::mem::size_of_val(model_albedo),
//         );

//         use itertools::izip;

//         for (model_mat, modelinvt_mat, albedo) in izip!(
//             model_matrices.iter().copied(),
//             modelinvt_mats,
//             model_albedo.iter().copied()
//         ) {
//             instance_data.extend_from_slice(bytemuck::cast_slice(&[model_mat]));
//             instance_data.extend_from_slice(bytemuck::cast_slice(&[modelinvt_mat]));
//             instance_data.extend_from_slice(bytemuck::cast_slice(&[albedo]));
//         }

//         device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//             label: None,
//             contents: bytemuck::cast_slice(&instance_data),
//             usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
//         })
//     }

//     pub fn new(device: &wgpu::Device, world_model: WorldModel) -> Self {
//         let WorldModel {
//             model,
//             model_matrices,
//             model_albedo,
//         } = world_model;

//         let gpu_model = GpuModel::new(device, model);
//         let instance_buf = Self::construct_instance_buf(device, &model_matrices, &model_albedo);

//         Self {
//             gpu_model,
//             model_albedo,
//             model_matrices,
//             instance_buf,
//         }
//     }

//     pub fn update_world(&mut self, queue: &wgpu::Queue, idx: usize, matrix: Matrix4<f32>) {
//         self.model_matrices[idx] = matrix;

//         let offset =
//             idx * (2 * std::mem::size_of::<Matrix4<f32>>() + std::mem::size_of::<Vector3<f32>>());

//         let mut updated_data: Vec<u8> = Vec::with_capacity(
//             2 * std::mem::size_of::<Matrix4<f32>>() + std::mem::size_of::<Vector3<f32>>(),
//         );

//         let model_mat = self.model_matrices[idx];
//         let modelinvt_mat = model_mat.try_inverse().unwrap().transpose();
//         let albedo = self.model_albedo[idx];

//         updated_data.extend_from_slice(bytemuck::cast_slice(&[model_mat]));
//         updated_data.extend_from_slice(bytemuck::cast_slice(&[modelinvt_mat]));
//         updated_data.extend_from_slice(bytemuck::cast_slice(&[albedo]));

//         queue.write_buffer(
//             &self.instance_buf,
//             offset as wgpu::BufferAddress,
//             bytemuck::cast_slice(&updated_data),
//         );
//     }

//     pub fn model_mat(&self, idx: usize) -> Matrix4<f32> {
//         self.model_matrices[idx]
//     }

//     fn num_instances(&self) -> u32 {
//         self.model_matrices.len() as u32
//     }

//     pub fn draw<'rpass, 'model: 'rpass>(&'model self, render_pass: &mut wgpu::RenderPass<'rpass>) {
//         self.gpu_model.configure_pass(render_pass);
//         render_pass.set_vertex_buffer(1, self.instance_buf.slice(..));

//         if self.gpu_model.indexed() {
//             render_pass.draw_indexed(0..self.gpu_model.num_indices(), 0, 0..self.num_instances());
//         } else {
//             render_pass.draw(0..self.gpu_model.num_indices(), 0..self.num_instances());
//         }
//     }
// }
