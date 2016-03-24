/// Convenient inference helper to export the interface without having to use explicit types.

use buffer::{Buffer, HasBuffer};
use core::marker::PhantomData;
use shader::program::{HasProgram, Program};
use shader::stage::{HasStage, Stage, StageError, ShaderTypeable};
use tessellation;
use vertex::Vertex;

pub struct Device<T>(PhantomData<T>);

impl<C> Device<C> where C: HasBuffer {
  pub fn new_buffer<A, T>(a: A, size: u32) -> Buffer<C, A, T> {
    Buffer::new(a, size)
  }
}

impl<C> Device<C> where C: tessellation::HasTessellation {
  pub fn new_tessellation<T>(mode: tessellation::Mode, vertices: Vec<T>, indices: Option<u32>) -> C::Tessellation where T: Vertex {
    C::new(mode, vertices, indices)
  }
}

impl<C> Device<C> where C: HasStage {
  pub fn new_stage<'a, 'b, T>(src: &'a str) -> Result<Stage<C, T>, StageError<'b>> where T: ShaderTypeable {
    Stage::new(src)
  }
}

//impl<C> Device<C> where C: HasProgram {
//  pub fn new_program(tess: Option<(&C::AStage, &C::AStage)>, vertex: &C::AStage, geometry: Option<&C::AStage>, fragment: &C::AStage) -> C::Program {
//    C::new(tess, vertex, geometry, fragment)
//  }
//}
