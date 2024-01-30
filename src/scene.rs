use nalgebra as na;

pub trait SceneMesh {
    fn layout(&self) -> wgpu::VertexBufferLayout<'static>;
    fn buffer(&self) -> wgpu::Buffer;
    fn bind_group(&self) -> Option<&wgpu::BindGroup>;
}

// TODO: Implement scene graph.
// Features:
// * Every scene object is constructed from N sub-meshes.
// * Every sub-mesh has its own local transform.
// * Every object has full model transform.
// * Every object can have their own bind group.
// * Scene is generic - it is parametrized by type O which implements SceneMesh.
// * It is responsibility of a caller to assign proper Scene type to proper pass.
