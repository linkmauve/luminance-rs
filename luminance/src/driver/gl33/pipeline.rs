use crate::driver::PipelineDriver;
use crate::driver::gl33::GL33;
use crate::driver::gl33::state::GraphicsState;
use std::cell::RefCell;
use std::rc::Rc;

// A stack of bindings.
//
// This type implements a stacking system for effective resource bindings by allocating new
// bindings points only when no recycled resource is available. It helps have a better memory
// footprint in the resource space.
struct BindingStack {
  gfx_state: Rc<RefCell<GraphicsState>>,
  next_texture_unit: u32,
  free_texture_units: Vec<u32>,
  next_buffer_binding: u32,
  free_buffer_bindings: Vec<u32>,
}

impl BindingStack {
  // Create a new, empty binding stack.
  fn new(gfx_state: Rc<RefCell<GraphicsState>>) -> Self {
    BindingStack {
      gfx_state,
      next_texture_unit: 0,
      free_texture_units: Vec::new(),
      next_buffer_binding: 0,
      free_buffer_bindings: Vec::new(),
    }
  }
}
