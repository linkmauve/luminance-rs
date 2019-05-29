use gl;
use gl::types::*;
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use crate::driver::FramebufferDriver;
use crate::driver::gl33::GL33;
use crate::driver::gl33::state::GraphicsState;
use crate::driver::gl33::texture::opengl_target;
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
pub struct Framebuffer {
  handle: GLuint,
  renderbuffer: Option<GLuint>,
  state: Rc<RefCell<GraphicsState>>,
}

impl Framebuffer {
  #[inline]
  pub(crate) fn handle(&self) -> GLuint {
    self.handle
  }
}

unsafe impl FramebufferDriver for GL33 {
  type Framebuffer = Framebuffer;

  type Err = FramebufferError;

  unsafe fn back_buffer(&mut self, _: [u32; 2]) -> Result<Self::Framebuffer, Self::Err> {
    Ok(Framebuffer {
      handle: 0,
      renderbuffer: None
    })
  }

  unsafe fn new_framebuffer<L, D, CS, DS>(
    &mut self,
    size: D::Size,
    mipmaps: usize
  ) -> Result<(Self::Framebuffer, CS, DS), Self::Err>
  where CS: ColorSlot<L, D>,
        DS: DepthSlot<L, D>,
        L: Layerable,
        D: Dimensionable,
        D::Size: Copy {
    let mut handle: GLuint = 0;
    let color_formats = CS::color_formats();
    let depth_format = DS::depth_format();
    let target = opengl_target(L::layering(), D::dim());
    let mut textures = vec![0; color_formats.len() + if depth_format.is_some() { 1 } else { 0 }];
    let mut depth_texture: Option<GLuint> = None;
    let mut depth_renderbuffer: Option<GLuint> = None;

    gl::GenFramebuffers(1, &mut handle);

    self.state.borrow_mut().bind_draw_framebuffer(handle);

    // generate all the required textures once; the textures vec will be reduced and dispatched
    // into other containers afterwards (in ColorSlot::reify_textures)
    gl::GenTextures((textures.len()) as GLint, textures.as_mut_ptr());

    // color textures
    if color_formats.is_empty() {
      gl::DrawBuffer(gl::NONE);
    } else {
      for (i, (format, texture)) in color_formats.iter().zip(&textures).enumerate() {
        self.state.borrow_mut().bind_texture(target, *texture);

        create_texture::<L, D>(target, size, mipmaps, *format, &Default::default())
          .map_err(FramebufferError::TextureError)?;
        gl::FramebufferTexture(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0 + i as GLenum, *texture, 0);
      }

      // specify the list of color buffers to draw to
      let color_buf_nb = color_formats.len() as GLsizei;
      let color_buffers: Vec<_> =
        (gl::COLOR_ATTACHMENT0..gl::COLOR_ATTACHMENT0 + color_buf_nb as GLenum).collect();

      gl::DrawBuffers(color_buf_nb, color_buffers.as_ptr());
    }

    // depth texture, if exists
    if let Some(format) = depth_format {
      let texture = textures.pop().unwrap();

      ctx.state().borrow_mut().bind_texture(target, texture);
      create_texture::<L, D>(target, size, mipmaps, format, &Default::default())
        .map_err(FramebufferError::TextureError)?;

      gl::FramebufferTexture(gl::FRAMEBUFFER, gl::DEPTH_ATTACHMENT, texture, 0);

      depth_texture = Some(texture);
    } else {
      let mut renderbuffer: GLuint = 0;

      gl::GenRenderbuffers(1, &mut renderbuffer);
      gl::BindRenderbuffer(gl::RENDERBUFFER, renderbuffer);
      gl::RenderbufferStorage(
        gl::RENDERBUFFER,
        gl::DEPTH_COMPONENT32F,
        D::width(size) as GLsizei,
        D::height(size) as GLsizei,
        );

      gl::BindRenderbuffer(gl::RENDERBUFFER, 0); // FIXME: see whether really needed

      gl::FramebufferRenderbuffer(
        gl::FRAMEBUFFER,
        gl::DEPTH_ATTACHMENT,
        gl::RENDERBUFFER,
        renderbuffer,
        );

      depth_renderbuffer = Some(renderbuffer);
    }

    self.state.borrow_mut().bind_texture(target, 0); // FIXME: see whether really needed

    let framebuffer = Framebuffer {
      handle,
      renderbuffer: depth_renderbuffer,
      state: self.state.clone(),
    };
    let cs = CS::reify_textures(ctx, size, mipmaps, &mut textures.into_iter());
    let ds = DS::reify_texture(ctx, size, mipmaps, depth_texture);

    match get_status() {
      Ok(_) => {
        ctx.state().borrow_mut().bind_draw_framebuffer(0); // FIXME: see whether really needed

        Ok((framebuffer, cs, ds))
      }
      Err(reason) => {
        ctx.state().borrow_mut().bind_draw_framebuffer(0); // FIXME: see whether really needed
        Self::drop_framebuffer(&mut framebuffer);
        Err(FramebufferError::Incomplete(reason))
      }
    }
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

fn get_status() -> Result<(), IncompleteReason> {
  let status = unsafe { gl::CheckFramebufferStatus(gl::FRAMEBUFFER) };

  match status {
    gl::FRAMEBUFFER_COMPLETE => Ok(()),
    gl::FRAMEBUFFER_UNDEFINED => Err(IncompleteReason::Undefined),
    gl::FRAMEBUFFER_INCOMPLETE_ATTACHMENT => Err(IncompleteReason::IncompleteAttachment),
    gl::FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT => Err(IncompleteReason::MissingAttachment),
    gl::FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER => Err(IncompleteReason::IncompleteDrawBuffer),
    gl::FRAMEBUFFER_INCOMPLETE_READ_BUFFER => Err(IncompleteReason::IncompleteReadBuffer),
    gl::FRAMEBUFFER_UNSUPPORTED => Err(IncompleteReason::Unsupported),
    gl::FRAMEBUFFER_INCOMPLETE_MULTISAMPLE => Err(IncompleteReason::IncompleteMultisample),
    gl::FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS => Err(IncompleteReason::IncompleteLayerTargets),
    _ => panic!("unknown OpenGL framebuffer incomplete status! status={}", status),
  }
}
