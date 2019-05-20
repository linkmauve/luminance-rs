use crate::driver::PipelineDriver;
use crate::driver::gl33::GL33;
use crate::driver::gl33::state::GraphicsState;
use std::cell::RefCell;
use std::fmt;
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

/// An opaque type used to create pipelines.
#[derive(Debug)]
struct Builder {
  binding_stack: Rc<RefCell<BindingStack>>,
}

#[derive(Debug)]
enum PipelineError {
}

impl fmt::Display for PipelineError {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    Ok(())
  }
}

unsafe impl PipelineDriver for GL33 {
  type Builder = Builder;

  type Err = PipelineError;

  unsafe fn new_builder(&mut self) -> Result<Self::Builder, <Self as PipelineDriver>::Err> {
    Ok(Builder {
      binding_stack: Rc::new(RefCell::new(BindingStack::new(gfx_state))),
    })
  }

  unsafe fn run_pipeline<F>(
    builder: &mut Self::Builder,
    framebufer: &mut Self::Framebuffer,
    framebufer_width: usize,
    framebufer_height: usize,
    clear_color: [f32; 4]
  ) where F: FnOnce(Self::Pipeline, Self::ShadingGate) {
    let binding_stack = &builder.binding_stack;

    unsafe {
      let bs = binding_stack.borrow();
      bs.gfx_state
        .borrow_mut()
        .bind_draw_framebuffer(framebuffer.handle());

      gl::Viewport(0, 0, framebuffer_width as GLint, framebuffer_height as GLint);
      gl::ClearColor(clear_color[0], clear_color[1], clear_color[2], clear_color[3]);
      gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
    }

    let p = Pipeline { binding_stack };
    let shd_gt = ShadingGate { binding_stack };

    f(p, shd_gt);
  }
}
