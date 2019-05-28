use gl;
use gl::types::*;
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use crate::driver::FramebufferDriver;
use crate::driver::gl33::GL33;
use crate::driver::gl33::state::GraphicsState;
use crate::framebuffer::{ColorSlot, DepthSlot};
use crate::texture::{Dimensionable, Layerable};

/// Framebuffer error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FramebufferError {
  TextureError(TextureError),
  Incomplete(IncompleteReason),
}

impl fmt::Display for FramebufferError {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      FramebufferError::TextureError(ref e) => write!(f, "framebuffer texture error: {}", e),

      FramebufferError::Incomplete(ref e) => write!(f, "incomplete framebuffer: {}", e),
    }
  }
}

/// Reason a framebuffer is incomplete.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IncompleteReason {
  Undefined,
  IncompleteAttachment,
  MissingAttachment,
  IncompleteDrawBuffer,
  IncompleteReadBuffer,
  Unsupported,
  IncompleteMultisample,
  IncompleteLayerTargets,
}

impl fmt::Display for IncompleteReason {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      IncompleteReason::Undefined => write!(f, "incomplete reason"),
      IncompleteReason::IncompleteAttachment => write!(f, "incomplete attachment"),
      IncompleteReason::MissingAttachment => write!(f, "missing attachment"),
      IncompleteReason::IncompleteDrawBuffer => write!(f, "incomplete draw buffer"),
      IncompleteReason::IncompleteReadBuffer => write!(f, "incomplete read buffer"),
      IncompleteReason::Unsupported => write!(f, "unsupported"),
      IncompleteReason::IncompleteMultisample => write!(f, "incomplete multisample"),
      IncompleteReason::IncompleteLayerTargets => write!(f, "incomplete layer targets"),
    }
  }
}

// OpenGL representation of a framebuffer.
pub struct RawFramebuffer {
  handle: GLuint,
  renderbuffer: Option<GLuint>,
  state: Rc<RefCell<GraphicsState>>,
}

unsafe impl FramebufferDriver for GL33 {
  type Framebuffer = RawFramebuffer;

  type Err = FramebufferError;

  unsafe fn back_buffer(&mut self, _: [u32; 2]) -> Result<Self::Framebuffer, Self::Err> {
    Ok(RawFramebuffer {
      handle: 0,
      renderbuffer: None
    })
  }

  unsafe fn new_framebuffer<L, D, CS, DS>(
    &mut self,
    size: [u32; 2],
    mipmaps: usize
  ) -> Result<Self::Framebuffer, Self::Err>
  where CS: ColorSlot<L, D>,
        DS: DepthSlot<L, D>,
        L: Layerable,
        D: Dimensionable,
        D::Size: Copy {
    unimplemented!()
  }

  unsafe fn drop_framebuffer(framebuffer: &mut Self::Framebuffer) {
    if let Some(ref renderbuffer) = framebuffer.renderbuffer {
      gl::DeleteRenderbuffers(1, renderbuffer);
    }

    if framebuffer.handle != 0 {
      gl::DeleteFramebuffers(1, &framebuffer.handle);
    }
  }

  unsafe fn use_framebuffer(framebuffer: &mut Self::Framebuffer) {
    framebuffer.state.borrow_mut().bind_draw_framebuffer(framebuffer.handle);
  }

  unsafe fn set_framebuffer_viewport(
    _: &mut Self::Framebuffer,
    x: u32,
    y: u32,
    width: u32,
    height: u32
  ) {
    gl::Viewport(x as GLint, y as GLint, width as GLsizei, height as GLsizei);
  }

  unsafe fn set_framebuffer_clear_color(
    _: &mut Self::Framebuffer,
    rgba: [f32; 4]
  ) {
    gl::ClearColor(
      rgba[0] as GLfloat,
      rgba[1] as GLfloat,
      rgba[2] as GLfloat,
      rgba[3] as GLfloat
    );
  }

  unsafe fn clear_framebuffer(_: &mut Self::Framebuffer) {
    gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
  }
}
