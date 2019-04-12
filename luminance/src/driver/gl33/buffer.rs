use crate::driver::BufferDriver;
use crate::driver::gl33::GL33;
use crate::driver::gl33::state::GraphicsState;
use gl;
use gl::types::*;
use std::cell::RefCell;
use std::mem;
use std::os::raw::c_void;
use std::ptr;
use std::rc::Rc;

#[derive(Debug)]
pub enum BufferDriverError {
  MapFailed
}

#[derive(Debug)]
pub struct RawBuffer {
  handle: GLuint,
  bytes: usize,
  len: usize,
  gfx_state: Rc<RefCell<GraphicsState>>
}

unsafe impl BufferDriver for GL33 {
  type Buffer = RawBuffer;

  type Err = BufferDriverError;

  unsafe fn new_buffer<T>(&mut self, len: usize) -> Result<Self::Buffer, Self::Err> {
    let mut buffer: GLuint = 0;
    let bytes = mem::size_of::<T>() * len;

    gl::GenBuffers(1, &mut buffer);
    self.state.borrow_mut().bind_array_buffer(buffer);
    gl::BufferData(gl::ARRAY_BUFFER, bytes as isize, ptr::null(), gl::STREAM_DRAW);

    Ok(RawBuffer {
      handle: buffer,
      bytes,
      len,
      gfx_state: self.state.clone(),
    })
  }

  unsafe fn from_slice<T>(&mut self, slice: &[T]) -> Result<Self::Buffer, Self::Err> {
    let mut buffer: GLuint = 0;
    let len = slice.len();
    let bytes = mem::size_of::<T>() * len;

    gl::GenBuffers(1, &mut buffer);
    self.state.borrow_mut().bind_array_buffer(buffer);
    gl::BufferData(
      gl::ARRAY_BUFFER,
      bytes as isize,
      slice.as_ptr() as *const c_void,
      gl::STREAM_DRAW,
    );

    Ok(RawBuffer {
      handle: buffer,
      bytes,
      len,
      gfx_state: self.state.clone(),
    })
  }

  unsafe fn drop(buffer: &mut Self::Buffer) {
    gl::DeleteBuffers(1, &mut buffer.handle)
  }

  unsafe fn at<T>(buffer: &Self::Buffer, i: usize) -> Option<T> where T: Copy {
    buffer.state.borrow_mut().bind_array_buffer(buffer.handle);

    let ptr = gl::MapBuffer(gl::ARRAY_BUFFER, gl::READ_ONLY) as *const T;
    let x = *ptr.offset(i as isize);
    let _ = gl::UnmapBuffer(gl::ARRAY_BUFFER);

    Some(x)
  }

  unsafe fn whole<T>(buffer: &Self::Buffer) -> Vec<T> {
    buffer.state.borrow_mut().bind_array_buffer(buffer.handle);

    let ptr = gl::MapBuffer(gl::ARRAY_BUFFER, gl::READ_ONLY) as *mut T;
    let len = buffer.len;
    let values = Vec::from_raw_parts(ptr, len, len);
    let _ = gl::UnmapBuffer(gl::ARRAY_BUFFER);

    values
  }

  unsafe fn set<T>(buffer: &mut Self::Buffer, i: usize, x: T) -> Result<(), Self::Err> {
    buffer.state.borrow_mut().bind_array_buffer(buffer.handle);

    let ptr = gl::MapBuffer(gl::ARRAY_BUFFER, gl::WRITE_ONLY) as *mut T;
    *ptr.offset(i as isize) = x;
    let _ = gl::UnmapBuffer(gl::ARRAY_BUFFER);

    Ok(())
  }

  unsafe fn write_whole<T>(buffer: &mut Self::Buffer, values: &[T], bytes: usize) -> Result<(), Self::Err> {
    buffer.state.borrow_mut().bind_array_buffer(buffer.handle);

    let ptr = gl::MapBuffer(gl::ARRAY_BUFFER, gl::WRITE_ONLY);
    ptr::copy_nonoverlapping(values.as_ptr() as *const c_void, ptr, bytes);
    let _ = gl::UnmapBuffer(gl::ARRAY_BUFFER);

    Ok(())
  }

  unsafe fn as_slice<T>(buffer: &Self::Buffer) -> Result<*const T, Self::Err> {
    buffer.state.borrow_mut().bind_array_buffer(buffer.handle);

    let ptr = gl::MapBuffer(gl::ARRAY_BUFFER, gl::READ_ONLY) as *const T;

    if ptr.is_null() {
      return Err(BufferDriverError::MapFailed);
    }

    Ok(ptr)
  }

  unsafe fn as_slice_mut<T>(buffer: &mut Self::Buffer) -> Result<*mut T, Self::Err> {
    buffer.state.borrow_mut().bind_array_buffer(buffer.handle);

    let ptr = gl::MapBuffer(gl::ARRAY_BUFFER, gl::READ_ONLY) as *mut T;

    if ptr.is_null() {
      return Err(BufferDriverError::MapFailed);
    }

    Ok(ptr)
  }

  unsafe fn drop_slice<T>(buffer: &mut Self::Buffer, _: *const T) {
    buffer.state.borrow_mut().bind_array_buffer(buffer.handle);
    gl::UnmapBuffer(gl::ARRAY_BUFFER);
  }

  unsafe fn drop_slice_mut<T>(buffer: &mut Self::Buffer, _: *mut T) {
    buffer.state.borrow_mut().bind_array_buffer(buffer.handle);
    gl::UnmapBuffer(gl::ARRAY_BUFFER);
  }
}
