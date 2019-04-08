use crate::driver::FramebufferDriver;
use crate::driver::gl33::GL33;
use crate::driver::gl33::state::GraphicsState;
use crate::framebuffer::{ColorSlot, DepthSlot};
use crate::texture::{
  create_texture, opengl_target, Dim2, Dimensionable, Flat, Layerable, RawTexture, Texture, TextureError,
};
use gl;
use gl::types::*;
use std::fmt;
use std::marker::PhantomData;

// OpenGL representation of a framebuffer.
pub struct RawFramebuffer {
    handle: GLuint,
    renderbuffer: Option<GLuint>
}

unsafe impl FramebufferDriver for GL33 {
  type Framebuffer = RawFramebuffer;

  type Err = (); // FIXME

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

  unsafe fn drop_framebuffer(&mut self, framebuffer: &mut Self::Framebuffer) {
    if let Some(renderbuffer) = framebuffer.renderbuffer {
      gl::DeleteRenderbuffers(1, &renderbuffer);
    }

    if framebuffer.handle != 0 {
      gl::DeleteFramebuffers(1, &framebuffer.handle);
    }
  }

  unsafe fn use_framebuffer(&mut self, framebuffer: &mut Self::Framebuffer) {
    self.state.borrow_mut().bind_draw_framebuffer(framebuffer.handle);
  }

  unsafe fn set_framebuffer_viewport(
    &mut self,
    framebuffer: &mut Self::Framebuffer,
    x: u32,
    y: u32,
    width: u32,
    height: u32
  ) {
    gl::Viewport(x as GLint, y as GLint, width as GLsizei, height as GLsizei);
  }

  unsafe fn set_framebuffer_clear_color(
    &mut self,
    framebuffer: &mut Self::Framebuffer,
    rgba: [f32; 4]
  ) {
    gl::ClearColor(
      rgba[0] as GLfloat,
      rgba[1] as GLfloat,
      rgba[2] as GLfloat,
      rgba[3] as GLfloat
    );
  }

  unsafe fn clear_framebuffer(&mut self, _: &mut Self::Framebuffer) {
    gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
  }
}
