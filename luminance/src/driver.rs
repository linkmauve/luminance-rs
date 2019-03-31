//! Graphics driver.
//!
//! A graphics driver is, as the name implies, an implementation of the graphics features that do
//! actual IO and effects. A driver typically implements “a technology” — e.g. OpenGL, Vulkan,
//! software-renderer, etc.

pub mod gl33;

use crate::blending;
use crate::depth_test;
use crate::face_culling;
use crate::pixel;
use crate::texture;
use crate::vertex_restart;

/// Main driver, providing all graphics-related features.
pub unsafe trait Driver: BufferDriver + RenderStateDriver {}

/// Buffer implementation.
pub unsafe trait BufferDriver {
  /// Representation of graphics buffers by this driver.
  type Buffer;

  /// Error that might occur with buffers.
  type Err;

  /// Create a new buffer with uninitialized memory.
  unsafe fn new_buffer<T>(&mut self, len: usize) -> Result<Self::Buffer, Self::Err>;
  /// Create a new buffer from a slice.
  unsafe fn from_slice<T>(&mut self, slice: &[T]) -> Result<Self::Buffer, Self::Err>;
  /// Drop a buffer.
  unsafe fn drop(&mut self, buffer: &mut Self::Buffer);
  /// Retrieve an element via indexing.
  unsafe fn at<T>(&mut self, buffer: &Self::Buffer, i: usize) -> Option<T> where T: Copy;
  /// Retrieve the whole content.
  unsafe fn whole<T>(&mut self, buffer: &Self::Buffer, len: usize) -> Vec<T>;
  /// Set a value at a given index.
  unsafe fn set<T>(&mut self, buffer: &mut Self::Buffer, i: usize, x: T) -> Result<(), Self::Err>;
  /// Write a whole slice into a buffer.
  unsafe fn write_whole<T>(&self, buffer: &mut Self::Buffer, values: &[T], bytes: usize) -> Result<(), Self::Err>;
  /// Obtain an immutable slice view into the buffer.
  unsafe fn as_slice<T>(&mut self, buffer: &Self::Buffer) -> Result<*const T, Self::Err>;
  /// Obtain an immutable slice view into the buffer.
  unsafe fn as_slice_mut<T>(&mut self, buffer: &mut Self::Buffer) -> Result<*mut T, Self::Err>;
  // Drop a slice.
  unsafe fn drop_slice<T>(&mut self, buffer: &mut Self::Buffer, slice: *const T);
  // Drop a mutable slice.
  unsafe fn drop_slice_mut<T>(&mut self, buffer: &mut Self::Buffer, slice: *mut T);
}

/// Render state implementation.
pub unsafe trait RenderStateDriver {
  /// Set the blending state.
  unsafe fn set_blending_state(&mut self, state: blending::BlendingState);
  /// Set the blending equation.
  unsafe fn set_blending_equation(&mut self, equation: blending::Equation);
  /// Set the blending function.
  unsafe fn set_blending_func(&mut self, src: blending::Factor, dest: blending::Factor);
  /// Set the depth test.
  unsafe fn set_depth_test(&mut self, depth_test: depth_test::DepthTest);
  /// Set the face culling state.
  unsafe fn set_face_culling_state(&mut self, state: face_culling::FaceCullingState);
  /// Set the face culling order.
  unsafe fn set_face_culling_order(&mut self, order: face_culling::FaceCullingOrder);
  /// Set the face culling mode.
  unsafe fn set_face_culling_mode(&mut self, mode: face_culling::FaceCullingMode);
}

/// Framebuffer implementation.
pub unsafe trait FramebufferDriver {
  /// Representation of a graphics framebuffer by this driver.
  type Framebuffer;

  /// Representation of a graphics renderbuffer by this driver.
  type RenderBuffer;

  /// Error that might occur with framebuffers.
  type Err;

  /// Get the back buffer, if any available.
  unsafe fn back_buffer(&mut self) -> Result<Self::Framebuffer, Self::Err>;
  /// Create a framebuffer.
  unsafe fn new_framebuffer(&mut self) -> Result<Self::Framebuffer, Self::Err>;
  /// Drop a framebuffer.
  unsafe fn drop_framebuffer(&mut self, framebuffer: &mut Self::Framebuffer);
}

/// Texture implementation.
pub unsafe trait TextureDriver {
  /// Representation of a graphics texture by this driver.
  type Texture;

  /// Error that might occur with textures.
  type Err;

  /// Create a new texture.
  unsafe fn new_texture<L, D, P>(
    &mut self,
    size: D::Size,
    mipmaps: usize,
    sampler: &texture::Sampler
  ) -> Result<Self::Texture, Self::Err>
  where L: texture::Layerable,
        D: texture::Dimensionable,
        P: pixel::Pixel;

  /// Drop a texture.
  unsafe fn drop_texture(&mut self, texture: &mut Self::Texture);

  /// Upload texels to a part of a texture.
  unsafe fn upload_part<L, D, P>(
    &mut self,
    texture: &Self::Texture,
    gen_mipmaps: bool,
    offset: D::Offset,
    size: D::Size,
    texels: &[P::Encoding]
  ) -> Result<(), Self::Err>
  where L: texture::Layerable,
        D: texture::Dimensionable,
        P: pixel::Pixel;

  /// Upload raw texels to a part of a texture.
  unsafe fn upload_part_raw<L, D, P>(
    &mut self,
    texture: &Self::Texture,
    gen_mipmaps: bool,
    offset: D::Offset,
    size: D::Size,
    texels: &[P::RawEncoding]
  ) -> Result<(), Self::Err>
  where L: texture::Layerable,
        D: texture::Dimensionable,
        P: pixel::Pixel;

  /// Get the raw texels associated with this texture.
  unsafe fn get_raw_texels<P>(
    &mut self,
    texture: &Self::Texture
  ) -> Result<Vec<P::RawEncoding>, Self::Err>
  where P: pixel::Pixel,
        P::RawEncoding: Copy;
}
