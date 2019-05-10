//! Tessellation features.
//!
//! # Tessellation mode
//!
//! Tessellation is geometric information. Currently, several kinds of tessellation are supported:
//!
//! - *point clouds*;
//! - *lines*;
//! - *line strips*;
//! - *triangles*;
//! - *triangle fans*;
//! - *triangle strips*.
//!
//! Those kinds of tessellation are designated by the `Mode` type.
//!
//! # Tessellation creation
//!
//! Creation is done via the [`Tess::new`] function. This function is polymorphing in the type of
//! vertices you send. See the [`TessVertices`] type for further details.
//!
//! ## On interleaved and deinterleaved vertices
//!
//! Because [`Tess::new`] uses your user-defined vertex type, it uses interleaved memory. That
//! means that all vertices are spread out in a single GPU memory region (a single buffer). This
//! behavior is fine for most applications that will want their shaders to use all vertex attributes
//! but sometimes you want a more specific memory strategy. For instance, some shaders won’t use all
//! of the available vertex attributes.
//!
//! [`Tess`] supports such situations with the [`Tess::new_deinterleaved`] method, that creates a
//! tessellation by lying vertex attributes out in their own respective buffers. The implication is
//! that the interface requires you to pass already deinterleaved vertices. Those are most of the
//! time isomorphic to tuples of slices.
//!
//! # Tessellation vertices CPU mapping
//!
//! It’s possible to map `Tess`’ vertices into your code. You’re provided with two types to do so:
//!
//! - [`BufferSlice`], which gives you an immutable access to the vertices.
//! - [`BufferSliceMut`], which gives you a mutable access to the vertices.
//!
//! You can retrieve those slices with the [`Tess::as_slice`] and [`Tess::as_slice_mut`] methods.
//!
//! # Tessellation render
//!
//! In order to render a [`Tess`], you have to use a [`TessSlice`] object. You’ll be able to use
//! that object in *pipelines*. See the `pipeline` module for further details.
//!
//! [`BufferSlice`]: crate/buffer/struct.BufferSlice.html
//! [`BufferSliceMut`]: crate/buffer/struct.BufferSliceMut.html
//! [`Tess`]: struct.Tess.html
//! [`Tess::as_slice`]: struct.Tess.html#method.as_slice
//! [`Tess::as_slice_mut`]: struct.Tess.html#method.as_slice_mut
//! [`Tess::new`]: struct.Tess.html#method.new
//! [`Tess::new_deinterleaved`]: struct.Tess.html#method.new_deinterleaved
//! [`TessSlice`]: struct.TessSlice.html

use std::fmt;
use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

use crate::buffer::{BufferError, BufferSlice, BufferSliceMut};
use crate::driver::{BufferDriver, TessDriver};
use crate::context::GraphicsContext;
use crate::vertex::{
  VertexBufferDesc, Vertex, VertexAttribDim, VertexAttribDesc, VertexAttribType, VertexDesc,
  VertexInstancing
};

/// Vertices can be connected via several modes.
#[derive(Copy, Clone, Debug)]
pub enum Mode {
  /// A single point.
  Point,
  /// A line, defined by two points.
  Line,
  /// A strip line, defined by at least two points and zero or many other ones.
  LineStrip,
  /// A triangle, defined by three points.
  Triangle,
  /// A triangle fan, defined by at least three points and zero or many other ones.
  TriangleFan,
  /// A triangle strip, defined by at least three points and zero or many other ones.
  TriangleStrip,
}

/// Error that can occur while trying to map GPU tessellation to host code.
#[derive(Debug, Eq, PartialEq)]
pub enum TessMapError<D> where D: BufferDriver {
  /// The CPU mapping failed due to buffer errors.
  VertexBufferMapFailed(BufferError<D>),
  /// Target type is not the same as the one stored in the buffer.
  TypeMismatch(VertexDesc, VertexDesc),
  /// The CPU mapping failed because you cannot map an attributeless tessellation since it doesn’t
  /// have any vertex attribute.
  ForbiddenAttributelessMapping,
  /// The CPU mapping failed because currently, mapping deinterleaved buffers is not supported via
  /// a single slice.
  ForbiddenDeinterleavedMapping,
}

impl<D> fmt::Display for TessMapError<D> where D: BufferDriver {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      TessMapError::VertexBufferMapFailed(ref e) => write!(f, "cannot map tessellation buffer: {}", e),

      TessMapError::TypeMismatch(ref a, ref b) =>
       write!(f, "cannot map tessellation: type mismatch between {:?} and {:?}", a, b),

      TessMapError::ForbiddenAttributelessMapping => f.write_str("cannot map an attributeless buffer"),

      TessMapError::ForbiddenDeinterleavedMapping => {
        f.write_str("cannot map a deinterleaved buffer as interleaved")
      }
    }
  }
}

struct VertexBuffer<D> where D: BufferDriver {
  /// Indexed format of the buffer.
  fmt: VertexDesc,
  /// Internal buffer.
  buf: D::Buffer,
}

enum TessBuilderError<D> where D: TessDriver {
  CannotCreate(<D as TessDriver>::Err)
}

/// Build tessellations the easy way.
pub struct TessBuilder<'a, C> where C: GraphicsContext, C::Driver: TessDriver {
  ctx: &'a mut C,
  inner: <C::Driver as TessDriver>::TessBuilder,
  restart_index: Option<u32>,
  mode: Mode,
  vert_nb: usize,
  inst_nb: usize,
}

impl<'a, C> TessBuilder<'a, C> where C: GraphicsContext, C::Driver: TessDriver {
  pub fn new(ctx: &'a mut C) -> Result<Self, TessBuilderError<C::Driver>> {
    let inner = ctx.driver().new_tess_builder().map_err(TessBuilderError::CannotCreate)?;

    Ok(TessBuilder {
      ctx,
      inner,
      restart_index: None,
      mode: Mode::Point,
      vert_nb: 0,
      inst_nb: 0,
    })
  }

  /// Add vertices to be part of the tessellation.
  ///
  /// This method can be used in several ways. First, you can decide to use interleaved memory, in
  /// which case you will call this method only once by providing an interleaved slice / borrowed
  /// buffer. Second, you can opt-in to use deinterleaved memory, in which case you will have
  /// several, smaller buffers of borrowed data and you will issue a call to this method for all of
  /// them.
  pub fn add_vertices<V, W>(mut self, vertices: W) -> Result<Self, TessError<C::Driver>> where W: AsRef<[V]>, V: Vertex {
    let vertices = vertices.as_ref();

    unsafe {
      self.ctx.driver()
        .add_vertices(&mut self.inner, vertices)
        .map_err(TessError::DriverError)?;
    }

    Ok(self)
  }

  pub fn add_instances<V, W>(mut self, instances: W) -> Result<Self, TessError<C::Driver>> where W: AsRef<[V]>, V: Vertex {
    let instances = instances.as_ref();

    unsafe {
      self.ctx.driver()
        .add_instances(&mut self.inner, instances)
        .map_err(TessError::DriverError)?;
    }

    Ok(self)
  }

  /// Set vertex indices in order to specify how vertices should be picked by the GPU pipeline.
  pub fn set_indices<T, I>(mut self, indices: T) -> Result<Self, TessError<C::Driver>> where T: AsRef<[I]>, I: TessIndex  {
    let indices = indices.as_ref();

    unsafe {
      self.ctx.driver()
        .set_indices(&mut self.inner, indices)
        .map_err(TessError::DriverError)?;
    }

    Ok(self)
  }

  pub fn set_mode(mut self, mode: Mode) -> Self {
    self.mode = mode;
    self
  }

  pub fn set_vertex_nb(mut self, nb: usize) -> Self {
    self.vert_nb = nb;
    self
  }

  pub fn set_instance_nb(mut self, nb: usize) -> Self {
    self.inst_nb = nb;
    self
  }

  /// Set the primitive restart index. The initial value is `None`, implying no primitive restart.
  pub fn set_primitive_restart_index(mut self, index: Option<u32>) -> Self {
    self.restart_index = index;
    self
  }

  pub fn build(self) -> Result<Tess<C::Driver>, TessError<C::Driver>> {
    unsafe {
      self.ctx.driver()
        .build_tess(self.inner, self.vert_nb, self.inst_nb)
        .map_err(TessError::DriverError)
    }
  }
}

pub enum TessError<D> where D: TessDriver {
  DriverError(<D as TessDriver>::Err),
  AttributelessError(String),
  LengthIncoherency(usize),
  Overflow(usize, usize),
}

/// Possible tessellation index types.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TessIndexType {
  U8,
  U16,
  U32,
}

/// Class of tessellation indexes.
///
/// Values which types implement this trait are allowed to be used to index tessellation in *indexed
/// draw commands*.
///
/// You shouldn’t have to worry to much about that trait. Have a look at the current implementors
/// for an exhaustive list of types you can use.
///
/// > Implementing this trait is `unsafe`.
pub unsafe trait TessIndex {
  const INDEX_TYPE: TessIndexType;
}

unsafe impl TessIndex for u8 {
  const INDEX_TYPE: TessIndexType = TessIndexType::U8;
}

unsafe impl TessIndex for u16 {
  const INDEX_TYPE: TessIndexType = TessIndexType::U16;
}

unsafe impl TessIndex for u32 {
  const INDEX_TYPE: TessIndexType = TessIndexType::U32;
}

pub struct Tess<D> where D: TessDriver {
  inner: D::Tess,
}

impl<D> Tess<D> where D: TessDriver {
  fn render<C>(&self, ctx: &mut C, start_index: usize, vert_nb: usize, inst_nb: usize)
  where C: GraphicsContext<Driver = D> {
    ctx.driver().render_tess(self, start_index, vert_nb, inst_nb)
  }

  pub fn as_slice<'a, V>(&'a self) -> Result<BufferSlice<V, D>, TessMapError<D>>
  where V: Vertex {
    let buf = D::tess_vertex_buffer(&self.inner).map_err(TessMapError::DriverError)?;
    BufferSlice::from_driver_buf_ref(buf).map_err(TessMapError::VertexBufferMapFailed)
  }

  pub fn as_slice_mut<'a, V>(&mut self) -> Result<BufferSliceMut<V, D>, TessMapError<D>>
  where V: Vertex {
    let buf = D::tess_vertex_buffer_mut(&mut self.inner).map_err(TessMapError::DriverError)?;
    BufferSliceMut::from_driver_buf_ref(buf).map_err(TessMapError::VertexBufferMapFailed)
  }

  pub fn as_inst_slice<'a, V>(&'a self) -> Result<BufferSlice<V, D>, TessMapError<D>>
  where V: Vertex {
    let buf = D::tess_inst_buffer(&mut self.inner).map_err(TessMapError::DriverError)?;
    BufferSlice::from_driver_buf_ref(buf).map_err(TessMapError::VertexBufferMapFailed)
  }

  pub fn as_inst_slice_mut<'a, V>(&mut self) -> Result<BufferSliceMut<V, D>, TessMapError<D>>
  where V: Vertex {
    let buf = D::tess_inst_buffer_mut(&mut self.inner).map_err(TessMapError::DriverError)?;
    BufferSliceMut::from_driver_buf_ref(buf).map_err(TessMapError::VertexBufferMapFailed)
  }
}

impl<D> Drop for Tess<D> where D: TessDriver {
  fn drop(&mut self) {
    unsafe { D::drop_tess(self) }
  }
}

/// Tessellation slice.
///
/// This type enables slicing a tessellation on the fly so that we can render patches of it.
#[derive(Clone)]
pub struct TessSlice<'a, D> where D: TessDriver {
  /// Tessellation to render.
  tess: &'a Tess<D>,
  /// Start index (vertex) in the tessellation.
  start_index: usize,
  /// Number of vertices to pick from the tessellation. If `None`, all of them are selected.
  vert_nb: usize,
  /// Number of instances to render.
  inst_nb: usize,
}

impl<'a, D> TessSlice<'a, D> where D: TessDriver {
  /// Create a tessellation render that will render the whole input tessellation with only one
  /// instance.
  pub fn one_whole(tess: &'a Tess<D>) -> Self {
    TessSlice {
      tess,
      start_index: 0,
      vert_nb: tess.vert_nb,
      inst_nb: tess.inst_nb,
    }
  }

  /// Create a tessellation render for a part of the tessellation starting at the beginning of its
  /// buffer with only one instance.
  ///
  /// The part is selected by giving the number of vertices to render.
  ///
  /// > Note: if you also need to use an arbitrary part of your tessellation (not starting at the
  /// > first vertex in its buffer), have a look at `TessSlice::one_slice`.
  ///
  /// # Panic
  ///
  /// Panic if the number of vertices is higher to the capacity of the tessellation’s vertex buffer.
  pub fn one_sub(tess: &'a Tess<D>, vert_nb: usize) -> Self {
    if vert_nb > tess.vert_nb {
      panic!(
        "cannot render {} vertices for a tessellation which vertex capacity is {}",
        vert_nb, tess.vert_nb
      );
    }

    TessSlice {
      tess,
      start_index: 0,
      vert_nb,
      inst_nb: 1,
    }
  }

  /// Create a tessellation render for a slice of the tessellation starting anywhere in its buffer
  /// with only one instance.
  ///
  /// The part is selected by giving the start vertex and the number of vertices to render. This
  ///
  /// # Panic
  ///
  /// Panic if the start vertex is higher to the capacity of the tessellation’s vertex buffer.
  ///
  /// Panic if the number of vertices is higher to the capacity of the tessellation’s vertex buffer.
  pub fn one_slice(tess: &'a Tess<D>, start: usize, nb: usize) -> Self {
    if start > tess.vert_nb {
      panic!(
        "cannot render {} vertices starting at vertex {} for a tessellation which vertex capacity is {}",
        nb, start, tess.vert_nb
      );
    }

    if nb > tess.vert_nb {
      panic!(
        "cannot render {} vertices for a tessellation which vertex capacity is {}",
        nb, tess.vert_nb
      );
    }

    TessSlice {
      tess,
      start_index: start,
      vert_nb: nb,
      inst_nb: 1,
    }
  }

  /// Render a tessellation.
  pub fn render<C>(&self, ctx: &mut C) where C: GraphicsContext<Driver = D> {
    self
      .tess
      .render(ctx, self.start_index, self.vert_nb, self.inst_nb);
  }
}

impl<'a, D> From<&'a Tess<D>> for TessSlice<'a, D> where D: TessDriver {
  fn from(tess: &'a Tess<D>) -> Self {
    TessSlice::one_whole(tess)
  }
}

pub trait TessSliceIndex<Idx, D> where D: TessDriver {
  fn slice<'a>(&'a self, idx: Idx) -> TessSlice<'a, D>;
}

impl<D> TessSliceIndex<RangeFull, D> for Tess<D> where D: TessDriver {
  fn slice<'a>(&self, _: RangeFull) -> TessSlice<'a, D> {
    TessSlice::one_whole(self)
  }
}

impl<D> TessSliceIndex<RangeTo<usize>, D> for Tess<D> where D: TessDriver {
  fn slice<'a>(&self, to: RangeTo<usize>) -> TessSlice<'a, D> {
    TessSlice::one_sub(self, to.end)
  }
}

impl<D> TessSliceIndex<RangeFrom<usize>, D> for Tess<D> where D: TessDriver {
  fn slice<'a>(&self, from: RangeFrom<usize>) -> TessSlice<'a, D> {
    TessSlice::one_slice(self, from.start, self.vert_nb)
  }
}

impl<D> TessSliceIndex<Range<usize>, D> for Tess<D> where D: TessDriver {
  fn slice<'a>(&self, range: Range<usize>) -> TessSlice<'a, D> {
    TessSlice::one_slice(self, range.start, range.end)
  }
}
