//! Graphics driver.
//!
//! A graphics driver is, as the name implies, an implementation of the graphics features that do
//! actual IO and effects. A driver typically implements “a technology” — e.g. OpenGL, Vulkan,
//! software-renderer, etc.

pub mod gl33;

use std::fmt::{Debug, Display};

use crate::blending;
use crate::depth_test;
use crate::face_culling;
use crate::framebuffer;
use crate::pixel;
use crate::shader::stage2;
use crate::tess;
use crate::texture;
use crate::vertex;

/// Main driver, providing all graphics-related features.
pub trait Driver: BufferDriver + RenderStateDriver + TextureDriver + FramebufferDriver + TessDriver {}

impl<T> Driver for T where T: ?Sized + BufferDriver + RenderStateDriver + TextureDriver + FramebufferDriver + TessDriver {}

/// Buffer implementation.
pub unsafe trait BufferDriver {
  /// Representation of graphics buffers by this driver.
  type Buffer;

  /// Error that might occur with buffers.
  type Err: Debug + Display;

  /// Create a new buffer with uninitialized memory.
  unsafe fn new_buffer<T>(&mut self, len: usize) -> Result<Self::Buffer, Self::Err>;

  /// Create a new buffer from a slice.
  unsafe fn from_slice<T>(&mut self, slice: &[T]) -> Result<Self::Buffer, Self::Err>;

  /// Get the length of the buffer.
  unsafe fn len(buffer: &Self::Buffer) -> usize;

  /// Get the number of bytes the buffer can accept.
  unsafe fn bytes(buffer: &Self::Buffer) -> usize;

  /// Drop a buffer.
  unsafe fn drop(buffer: &mut Self::Buffer);

  /// Retrieve an element via indexing.
  unsafe fn at<T>(buffer: &Self::Buffer, i: usize) -> Option<T> where T: Copy;

  /// Retrieve the whole content.
  unsafe fn whole<T>(buffer: &Self::Buffer) -> Vec<T>;

  /// Set a value at a given index.
  unsafe fn set<T>(buffer: &mut Self::Buffer, i: usize, x: T) -> Result<(), Self::Err>;

  /// Write a whole slice into a buffer.
  unsafe fn write_whole<T>(buffer: &mut Self::Buffer, values: &[T], bytes: usize) -> Result<(), Self::Err>;

  /// Obtain an immutable slice view into the buffer.
  unsafe fn as_slice<T>(buffer: &Self::Buffer) -> Result<*const T, Self::Err>;

  /// Obtain an immutable slice view into the buffer.
  unsafe fn as_slice_mut<T>(buffer: &mut Self::Buffer) -> Result<*mut T, Self::Err>;

  // Drop a slice.
  unsafe fn drop_slice<T>(buffer: &Self::Buffer, slice: *const T);

  // Drop a mutable slice.
  unsafe fn drop_slice_mut<T>(buffer: &Self::Buffer, slice: *mut T);
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

/// Texture implementation.
pub unsafe trait TextureDriver {
  /// Representation of a graphics texture by this driver.
  type Texture;

  /// Error that might occur with textures.
  type Err: Debug + Display;

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
  unsafe fn drop_texture(texture: &mut Self::Texture);

  /// Upload texels to a part of a texture.
  unsafe fn upload_part<L, D, P>(
    texture: &mut Self::Texture,
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
    texture: &mut Self::Texture,
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
    texture: &Self::Texture
  ) -> Result<Vec<P::RawEncoding>, Self::Err>
  where P: pixel::Pixel,
        P::RawEncoding: Copy;
}

/// Framebuffer implementation.
pub unsafe trait FramebufferDriver: TextureDriver {
  /// Representation of a graphics framebuffer by this driver.
  type Framebuffer;

  /// Error that might occur with framebuffers.
  type Err: Debug + Display;

  /// Get the back buffer, if any available.
  unsafe fn back_buffer(
    &mut self,
    size: [u32; 2]
  ) -> Result<Self::Framebuffer, <Self as FramebufferDriver>::Err>;

  /// Create a framebuffer.
  unsafe fn new_framebuffer<L, D, CS, DS>(
    &mut self,
    size: D::Size,
    mipmaps: usize
  ) -> Result<(Self::Framebuffer, CS::ColorTextures, DS::DepthTexture), <Self as FramebufferDriver>::Err>
  where CS: framebuffer::ColorSlot<Self, L, D>,
        DS: framebuffer::DepthSlot<Self, L, D>,
        L: texture::Layerable,
        D: texture::Dimensionable,
        D::Size: Copy;

  /// Drop a framebuffer.
  unsafe fn drop_framebuffer(framebuffer: &mut Self::Framebuffer);

  /// Use a framebuffer.
  unsafe fn use_framebuffer(framebuffer: &mut Self::Framebuffer);

  /// Set the viewport for incoming calls in this framebuffer.
  unsafe fn set_framebuffer_viewport(
    framebuffer: &mut Self::Framebuffer,
    x: u32,
    y: u32,
    width: u32,
    height: u32
  );

  /// Clear color to use with this framebuffer when clearing.
  unsafe fn set_framebuffer_clear_color(
    framebuffer: &mut Self::Framebuffer,
    rgba: [f32; 4]
  );

  /// Clear a framebuffer.
  unsafe fn clear_framebuffer(framebuffer: &mut Self::Framebuffer);
}

/// Tessellation implementation.
pub unsafe trait TessDriver: BufferDriver {
  /// Representation of a graphics tessellation by this driver.
  type Tess;

  /// Representation of a graphics tessellation builder by this driver.
  type TessBuilder;

  /// Error that might occur with tessellations.
  type Err: Debug + Display;

  /// Get the default number of vertices in a tessellation.
  unsafe fn vert_nb(tess: &Self::Tess) -> usize;

  /// Get the default number of instances in a tessellation.
  unsafe fn inst_nb(tess: &Self::Tess) -> usize;

  /// Create an empty tessellation builder.
  unsafe fn new_tess_builder(&mut self) -> Result<Self::TessBuilder, <Self as TessDriver>::Err>;

  /// Add vertices to a tessellation builder.
  unsafe fn add_vertices<V>(
    &mut self,
    builder: &mut Self::TessBuilder,
    vertices: &[V]
  ) -> Result<(), <Self as TessDriver>::Err>
  where V: vertex::Vertex;

  /// Add instances to a tessellation builder.
  unsafe fn add_instances<V>(
    &mut self,
    builder: &mut Self::TessBuilder,
    instances: &[V]
  ) -> Result<(), <Self as TessDriver>::Err>
  where V: vertex::Vertex;

  /// Set vertex indices in order to specify how vertices should be picked by the GPU pipeline.
  unsafe fn set_indices<I>(
    &mut self,
    builder: &mut Self::TessBuilder,
    indices: &[I]
  ) -> Result<(), <Self as TessDriver>::Err>
  where I: tess::TessIndex;

  /// Build a tessellation out of a tessellation builder.
  unsafe fn build_tess(
    &mut self,
    builder: Self::TessBuilder,
    vert_nb: usize,
    inst_nb: usize
  ) -> Result<Self::Tess, <Self as TessDriver>::Err>;

  /// Drop a tessellation.
  unsafe fn drop_tess(tess: &mut Self::Tess);

  /// Get the internal buffer of the tessellation’s vertices in read-only mode.
  unsafe fn tess_vertex_buffer<'a, V>(
    tess: &'a Self::Tess
  ) -> Result<&'a Self::Buffer, <Self as TessDriver>::Err>
  where V: vertex::Vertex;

  /// Get the internal buffer of the tessellation’s vertices in read-write mode.
  unsafe fn tess_vertex_buffer_mut<'a, V>(
    tess: &'a mut Self::Tess
  ) -> Result<&'a mut Self::Buffer, <Self as TessDriver>::Err>
  where V: vertex::Vertex;

  /// Get the internal buffer of the tessellation’s instances in read-only mode.
  unsafe fn tess_inst_buffer<'a, V>(
    tess: &'a Self::Tess
  ) -> Result<&'a Self::Buffer, <Self as TessDriver>::Err>
  where V: vertex::Vertex;

  /// Get the internal buffer of the tessellation’s instances in read-write mode.
  unsafe fn tess_inst_buffer_mut<'a, V>(
    tess: &'a mut Self::Tess
  ) -> Result<&'a mut Self::Buffer, <Self as TessDriver>::Err>
  where V: vertex::Vertex;

  /// Render a tessellation.
  unsafe fn render_tess(
    &mut self,
    tess: &Self::Tess,
    start_index: usize,
    vert_nb: usize,
    inst_nb: usize
  );
}

/// Shader implementation.
pub unsafe trait ShaderDriver {
  /// Representation of a shader stage.
  type Stage;

  /// Representation of a shader program.
  type Program;

  // /// Representation of a shader uniform.
  // type Uniform;

  /// Type of error that can occur in implementations.
  type Err: Debug + Display;

  /// Create a new shader stage based on a string representing its source code.
  unsafe fn new_shader_stage(ty: stage2::Type, src: &str) -> Result<Self::Stage, Self::Err>;

  /// Drop a shader stage.
  unsafe fn drop_shader_stage(stage: &mut Self::Stage);

  /// Source a shader stage with a source code.
  unsafe fn source_shader_stage(stage: &mut Self::Stage, src: &str) -> Result<(), Self::Err>;

  /// Create a new program by attaching shader stages.
  unsafe fn new_shader_program<'a, T, G>(
    tess: T,
    vertex: &'a Self::Stage,
    geometry: G,
    fragment: &'a Self::Stage
  ) -> Result<Self::Program, Self::Err>
  where T: Into<Option<(&'a Self::Stage, &'a Self::Stage)>>,
        G: Into<Option<&'a Self::Stage>>;

  /// Link a shader program.
  unsafe fn link_shader_program(program: &Self::Program) -> Result<(), Self::Err>;
}

// /// Rendering pipeline implementation.
// pub unsafe trait PipelineDriver: BufferDriver + FramebufferDriver + TextureDriver {
//   type Builder;
//
//   type Pipeline;
//
//   type ShadingGate;
//
//   type BoundTexture;
//
//   type BoundBuffer;
//
//   type Err: Debug + Display;
//
//   /// Create a new pipeline builder.
//   unsafe fn new_builder(&mut self) -> Result<Self::Builder, <Self as PipelineDriver>::Err>;
//
//   /// Run a pipeline.
//   unsafe fn run_pipeline<F>(
//     builder: &mut Self::Builder,
//     framebufer: &mut Self::Framebuffer,
//     framebufer_width: usize,
//     framebufer_height: usize,
//     clear_color: [f32; 4]
//   ) where F: FnOnce(Self::Pipeline, Self::ShadingGate);
//
//   /// Bind a texture and return a bound texture.
//   unsafe fn bind_texture(
//     pipeline: &mut Self::Pipeline,
//     texture: &Self::Texture
//   ) -> Result<Self::BoundTexture, <Self as PipelineDriver>::Err>;
//
//   /// Bind a buffer and return a bound buffer.
//   unsafe fn bind_buffer(
//     pipeline: &mut Self::Pipeline,
//     buffer: &Self::Buffer
//   ) -> Result<Self::BoundBuffer, <Self as PipelineDriver>::Err>;
//
//   /// Run a shader on a set of rendering commands.
//   unsafe fn shade<F>(
//     shading_gate: &mut Self::ShadingGate,
//     program: &Program,
//     f: F
//   ) where F: FnOnce(&Self::RenderGate,
// }
