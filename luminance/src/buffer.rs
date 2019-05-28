//! Static GPU typed arrays.
//!
//! A GPU buffer is a typed continuous region of data. It has a size and can hold several elements.
//!
//! Buffers are created with the `new` associated function. You pass in the number of elements you
//! want in the buffer along with the `GraphicsContext` to create the buffer in.
//!
//! ```ignore
//! let buffer: Buffer<f32> = Buffer::new(&mut ctx, 5);
//! ```
//!
//! Once the buffer is created, you can perform several operations on them:
//!
//! - Writing to them.
//! - Reading from them.
//! - Passing them around as uniforms.
//! - Etc.
//!
//! However, you cannot change their size at runtime.
//!
//! # Writing to a buffer
//!
//! `Buffer`s support several write methods. The simple one is *clearing*. That is, replacing the
//! whole content of the buffer with a single value. Use the `clear` function to do so.
//!
//! ```ignore
//! buffer.clear(0.);
//! ```
//!
//! If you want to clear the buffer by providing a value for each elements, you want *filling*. Use
//! the `fill` function:
//!
//! ```ignore
//! buffer.fill([1., 2., 3., 4., 5.]);
//! ```
//!
//! You want to change a value at a given index? Easy, you can use the `set` function:
//!
//! ```ignore
//! buffer.set(3, 3.14);
//! ```
//!
//! # Reading from the buffer
//!
//! You can either retrieve the `whole` content of the `Buffer` or `get` a value with an index.
//!
//! ```ignore
//! // get the whole content
//! let all_elems = buffer.whole();
//! assert_eq!(all_elems, vec![1., 2., 3., 3.14, 5.]); // admit floating equalities
//!
//! // get the element at index 3
//! assert_eq!(buffer.at(3), Some(3.14));
//! ```
//!
//! # Uniform buffer
//!
//! It’s possible to use buffers as *uniform buffers*. That is, buffers that will be in bound at
//! rendering time and which content will be available for a shader to read (no write).
//!
//! In order to use your buffers in a uniform context, the inner type has to implement
//! `UniformBlock`. Keep in mind alignment must be respected and is a bit peculiar. TODO: explain
//! std140 here.

use std::cmp::Ordering;
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::slice;

use crate::context::GraphicsContext;
use crate::driver::BufferDriver;
use crate::linear::{M22, M33, M44};

/// Buffer errors.
#[derive(Debug, Eq, PartialEq)]
pub enum BufferError<D> where D: BufferDriver {
  /// Error occurring in a driver.
  DriverError(D::Err),
  /// Overflow when setting a value with a specific index.
  ///
  /// Contains the index and the size of the buffer.
  Overflow(usize, usize),
  /// Too few values were passed to fill a buffer.
  ///
  /// Contains the number of passed value and the size of the buffer.
  TooFewValues(usize, usize),
  /// Too many values were passed to fill a buffer.
  ///
  /// Contains the number of passed value and the size of the buffer.
  TooManyValues(usize, usize),
  /// Mapping the buffer failed.
  MapFailed,
}

impl<D> fmt::Display for BufferError<D> where D: BufferDriver{
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      BufferError::DriverError(ref e) => write!(f, "buffer driver error: {}", e),

      BufferError::Overflow(i, size) => write!(f, "buffer overflow (index = {}, size = {})", i, size),

      BufferError::TooFewValues(nb, size) => write!(
        f,
        "too few values passed to the buffer (nb = {}, size = {})",
        nb,
        size
      ),

      BufferError::TooManyValues(nb, size) => write!(
        f,
        "too many values passed to the buffer (nb = {}, size = {})",
        nb,
        size
      ),

      BufferError::MapFailed => write!(f, "buffer mapping failed"),
    }
  }
}

/// A `Buffer` is a GPU region you can picture as an array. It has a static size and cannot be
/// resized. The size is expressed in number of elements lying in the buffer – not in bytes.
pub struct Buffer<T, D> where D: BufferDriver {
  buf: D::Buffer,
  _t: PhantomData<T>,
}

impl<T, D> Buffer<T, D> where D: BufferDriver {
  /// Create a new `Buffer` with a given number of elements.
  pub fn new<C>(ctx: &mut C, len: usize) -> Result<Self, BufferError<D>> where C: GraphicsContext<Driver = D> {
    unsafe {
      ctx.driver()
        .new_buffer::<T>(len)
        .map(|buf| Buffer { buf, _t: PhantomData })
        .map_err(BufferError::DriverError)
    }
  }

  /// Create a buffer out of a slice.
  pub fn from_slice<C>(ctx: &mut C, slice: &[T]) -> Result<Self, BufferError<D>> where C: GraphicsContext<Driver = D> {
    Self::from_slice_driver(ctx.driver(), slice)
  }

  /// Create a buffer out of a slice (driver version).
  pub(crate) fn from_slice_driver(driver: &mut D, slice: &[T]) -> Result<Self, BufferError<D>> {
    unsafe {
      driver
        .from_slice::<T>(slice)
        .map(|buf| Buffer { buf, _t: PhantomData })
        .map_err(BufferError::DriverError)
    }
  }

  /// Retrieve an element from the `Buffer`.
  ///
  /// Checks boundaries.
  pub fn at(&self, i: usize) -> Option<T> where T: Copy {
    if i >= unsafe { D::len(&self.buf) } {
      return None;
    }

    unsafe { D::at(&self.buf, i) }
  }

  /// Retrieve the whole content of the `Buffer`.
  pub fn whole(&self) -> Vec<T> where T: Copy {
    unsafe { D::whole(&self.buf) }
  }

  /// Set a value at a given index in the `Buffer`.
  ///
  /// Checks boundaries.
  pub fn set(&mut self, i: usize, x: T) -> Result<(), BufferError<D>> where T: Copy {
    if i >= unsafe { D::len(&self.buf) } {
      return Err(BufferError::Overflow(i, unsafe { D::len(&self.buf) }));
    }

    unsafe { D::set(&mut self.buf, i, x).map_err(BufferError::DriverError) }
  }

  /// Write a whole slice into a buffer.
  ///
  /// If the slice you pass in has less items than the length of the buffer, you’ll get a
  /// `BufferError::TooFewValues` error. If it has more, you’ll get `BufferError::TooManyValues`.
  ///
  /// This function won’t write anything on any error.
  pub fn write_whole(&mut self, values: &[T]) -> Result<(), BufferError<D>> {
    let buf_len = unsafe { D::len(&self.buf) };
    let len = values.len();
    let in_bytes = len * mem::size_of::<T>();

    // generate warning and recompute the proper number of bytes to copy
    unsafe {
      let real_bytes = match in_bytes.cmp(&D::bytes(&self.buf)) {
        Ordering::Less => return Err(BufferError::TooFewValues(len, buf_len)),
        Ordering::Greater => return Err(BufferError::TooManyValues(len, buf_len)),
        _ => in_bytes,
      };

      D::write_whole(&mut self.buf, values, real_bytes).map_err(BufferError::DriverError)
    }
  }

  /// Fill the `Buffer` with a single value.
  pub fn clear(&mut self, x: T) -> Result<(), BufferError<D>> where T: Copy {
    self.write_whole(&vec![x; unsafe { D::len(&self.buf) }])
  }

  /// Fill the whole buffer with an array.
  pub fn fill(&mut self, values: &[T]) -> Result<(), BufferError<D>> {
    self.write_whole(values)
  }

  /// Convert a buffer to its driver representation.
  ///
  /// Becareful: once you have called this function, it is not possible to go back to a `Buffer<_>`.
  pub(crate) fn to_driver_buf(mut self) -> D::Buffer {
    let buf = mem::replace(&mut self.buf, unsafe { mem::uninitialized() });

    // forget self so that we don’t call drop on it after the function has returned
    mem::forget(self);

    buf
  }

  /// Obtain an immutable slice view into the buffer.
  pub fn as_slice(&self) -> Result<BufferSlice<T, D>, BufferError<D>> {
    BufferSlice::from_driver_buf_ref(&self.buf)
  }

  /// Obtain a mutable slice view into the buffer.
  pub fn as_slice_mut(&mut self) -> Result<BufferSliceMut<T, D>, BufferError<D>> {
    BufferSliceMut::from_driver_buf_ref(&mut self.buf)
  }
}

impl<T, D> Deref for Buffer<T, D> where D: BufferDriver {
  type Target = D::Buffer;

  fn deref(&self) -> &Self::Target {
    &self.buf
  }
}

impl<T, D> DerefMut for Buffer<T, D> where D: BufferDriver {
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.buf
  }
}

impl<T, D> Drop for Buffer<T, D> where D: BufferDriver {
  fn drop(&mut self) {
    unsafe { D::drop(self) }
  }
}

/// A buffer slice mapped into GPU memory.
pub struct BufferSlice<'a, T, D> where T: 'a, D: BufferDriver {
  // Borrowed raw buffer.
  buf: &'a D::Buffer,
  // Raw pointer into the GPU memory.
  ptr: *const T,
}

impl<'a, T, D> BufferSlice<'a, T, D> where T: 'a, D: BufferDriver {
  /// Create a buffer slice from a driver’s buffer representation.
  pub(crate) fn from_driver_buf_ref(buf: &'a D::Buffer) -> Result<Self, BufferError<D>> {
    let ptr = unsafe { D::as_slice::<T>(buf).map_err(BufferError::DriverError)? };
    Ok(BufferSlice { buf, ptr })
  }
}

impl<'a, T, D> Drop for BufferSlice<'a, T, D> where T: 'a, D: BufferDriver {
  fn drop(&mut self) {
    unsafe { D::drop_slice(&mut self.buf, self.ptr) }
  }
}

impl<'a, T, D> Deref for BufferSlice<'a, T, D> where T: 'a, D: BufferDriver {
  type Target = [T];

  fn deref(&self) -> &Self::Target {
    unsafe { slice::from_raw_parts(self.ptr, D::len(&self.buf)) }
  }
}

impl<'a, 'b, T, D> IntoIterator for &'b BufferSlice<'a, T, D> where T: 'a, D: BufferDriver {
  type IntoIter = slice::Iter<'b, T>;
  type Item = &'b T;

  fn into_iter(self) -> Self::IntoIter {
    self.deref().into_iter()
  }
}

/// A buffer mutable slice into GPU memory.
pub struct BufferSliceMut<'a, T, D> where T: 'a, D: BufferDriver {
  // Borrowed buffer.
  buf: &'a D::Buffer,
  // Raw pointer into the GPU memory.
  ptr: *mut T,
}

impl<'a, T, D> BufferSliceMut<'a, T, D> where T: 'a, D: BufferDriver {
  /// Create a buffer slice from a driver’s buffer representation.
  pub(crate) fn from_driver_buf_ref(buf: &'a mut D::Buffer) -> Result<Self, BufferError<D>> {
    let ptr = unsafe { D::as_slice_mut::<T>(buf).map_err(BufferError::DriverError)? };
    Ok(BufferSliceMut { buf, ptr })
  }
}

impl<'a, T, D> Drop for BufferSliceMut<'a, T, D> where T: 'a, D: BufferDriver {
  fn drop(&mut self) {
    unsafe { D::drop_slice_mut(&mut self.buf, self.ptr) }
  }
}

impl<'a, 'b, T, D> IntoIterator for &'b BufferSliceMut<'a, T, D> where T: 'a, D: BufferDriver {
  type IntoIter = slice::Iter<'b, T>;
  type Item = &'b T;

  fn into_iter(self) -> Self::IntoIter {
    self.deref().into_iter()
  }
}

impl<'a, 'b, T, D> IntoIterator for &'b mut BufferSliceMut<'a, T, D> where T: 'a, D: BufferDriver {
  type IntoIter = slice::IterMut<'b, T>;
  type Item = &'b mut T;

  fn into_iter(self) -> Self::IntoIter {
    self.deref_mut().into_iter()
  }
}

impl<'a, T, D> Deref for BufferSliceMut<'a, T, D> where T: 'a, D: BufferDriver {
  type Target = [T];

  fn deref(&self) -> &Self::Target {
    unsafe { slice::from_raw_parts(self.ptr, D::len(&self.buf)) }
  }
}

impl<'a, T, D> DerefMut for BufferSliceMut<'a, T, D> where T: 'a, D: BufferDriver {
  fn deref_mut(&mut self) -> &mut Self::Target {
    unsafe { slice::from_raw_parts_mut(self.ptr, D::len(&self.buf)) }
  }
}

/// Typeclass of types that can be used inside a uniform block. You have to be extra careful when
/// using uniform blocks and ensure you respect the OpenGL *std140* alignment / size rules. This
/// will be fixed in a future release.
pub unsafe trait UniformBlock {}

unsafe impl UniformBlock for u8 {}
unsafe impl UniformBlock for u16 {}
unsafe impl UniformBlock for u32 {}

unsafe impl UniformBlock for i8 {}
unsafe impl UniformBlock for i16 {}
unsafe impl UniformBlock for i32 {}

unsafe impl UniformBlock for f32 {}
unsafe impl UniformBlock for f64 {}

unsafe impl UniformBlock for bool {}

unsafe impl UniformBlock for M22 {}
unsafe impl UniformBlock for M33 {}
unsafe impl UniformBlock for M44 {}

unsafe impl UniformBlock for [u8; 2] {}
unsafe impl UniformBlock for [u16; 2] {}
unsafe impl UniformBlock for [u32; 2] {}

unsafe impl UniformBlock for [i8; 2] {}
unsafe impl UniformBlock for [i16; 2] {}
unsafe impl UniformBlock for [i32; 2] {}

unsafe impl UniformBlock for [f32; 2] {}
unsafe impl UniformBlock for [f64; 2] {}

unsafe impl UniformBlock for [bool; 2] {}

unsafe impl UniformBlock for [u8; 3] {}
unsafe impl UniformBlock for [u16; 3] {}
unsafe impl UniformBlock for [u32; 3] {}

unsafe impl UniformBlock for [i8; 3] {}
unsafe impl UniformBlock for [i16; 3] {}
unsafe impl UniformBlock for [i32; 3] {}

unsafe impl UniformBlock for [f32; 3] {}
unsafe impl UniformBlock for [f64; 3] {}

unsafe impl UniformBlock for [bool; 3] {}

unsafe impl UniformBlock for [u8; 4] {}
unsafe impl UniformBlock for [u16; 4] {}
unsafe impl UniformBlock for [u32; 4] {}

unsafe impl UniformBlock for [i8; 4] {}
unsafe impl UniformBlock for [i16; 4] {}
unsafe impl UniformBlock for [i32; 4] {}

unsafe impl UniformBlock for [f32; 4] {}
unsafe impl UniformBlock for [f64; 4] {}

unsafe impl UniformBlock for [bool; 4] {}

unsafe impl<T> UniformBlock for [T] where T: UniformBlock {}

macro_rules! impl_uniform_block_tuple {
  ($( $t:ident ),*) => {
    unsafe impl<$($t),*> UniformBlock for ($($t),*) where $($t: UniformBlock),* {}
  }
}

impl_uniform_block_tuple!(A, B);
impl_uniform_block_tuple!(A, B, C);
impl_uniform_block_tuple!(A, B, C, D);
impl_uniform_block_tuple!(A, B, C, D, E);
impl_uniform_block_tuple!(A, B, C, D, E, F);
impl_uniform_block_tuple!(A, B, C, D, E, F, G);
impl_uniform_block_tuple!(A, B, C, D, E, F, G, H);
impl_uniform_block_tuple!(A, B, C, D, E, F, G, H, I);
impl_uniform_block_tuple!(A, B, C, D, E, F, G, H, I, J);
