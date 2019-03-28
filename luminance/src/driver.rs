//! Graphics driver.
//!
//! A graphics driver is, as the name implies, an implementation of the graphics features that do
//! actual IO and effects. A driver typically implements “a technology” — e.g. OpenGL, Vulkan,
//! software-renderer, etc.

pub mod gl33;

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
