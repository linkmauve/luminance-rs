mod state;

use crate::blending::{BlendingState, Equation, Factor};
use crate::depth_test::DepthTest;
use crate::driver::{BufferDriver, RenderStateDriver};
use crate::driver::gl33::state::GraphicsState;
use crate::face_culling::{FaceCullingMode, FaceCullingOrder, FaceCullingState};
use gl;
use gl::types::*;
use std::cell::RefCell;
use std::mem;
use std::os::raw::c_void;
use std::ptr;
use std::rc::Rc;

pub struct GL33 {
  state: Rc<RefCell<GraphicsState>>
}

#[derive(Debug)]
pub enum BufferDriverError {
  MapFailed
}

unsafe impl BufferDriver for GL33 {
  type Buffer = GLuint;

  type Err = BufferDriverError;

  unsafe fn new_buffer<T>(&mut self, len: usize) -> Result<Self::Buffer, Self::Err> {
    let mut buffer: GLuint = 0;
    let bytes = mem::size_of::<T>() * len;

    gl::GenBuffers(1, &mut buffer);
    self.state.borrow_mut().bind_array_buffer(buffer);
    gl::BufferData(gl::ARRAY_BUFFER, bytes as isize, ptr::null(), gl::STREAM_DRAW);

    Ok(buffer)
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

    Ok(buffer)
  }

  unsafe fn drop(&mut self, buffer: &mut Self::Buffer) {
    gl::DeleteBuffers(1, buffer)
  }

  unsafe fn at<T>(&mut self, buffer: &Self::Buffer, i: usize) -> Option<T> where T: Copy {
    self.state.borrow_mut().bind_array_buffer(*buffer);

    let ptr = gl::MapBuffer(gl::ARRAY_BUFFER, gl::READ_ONLY) as *const T;
    let x = *ptr.offset(i as isize);
    let _ = gl::UnmapBuffer(gl::ARRAY_BUFFER);

    Some(x)
  }

  unsafe fn whole<T>(&mut self, buffer: &Self::Buffer, len: usize) -> Vec<T> {
    self.state.borrow_mut().bind_array_buffer(*buffer);

    let ptr = gl::MapBuffer(gl::ARRAY_BUFFER, gl::READ_ONLY) as *mut T;
    let values = Vec::from_raw_parts(ptr, len, len);
    let _ = gl::UnmapBuffer(gl::ARRAY_BUFFER);

    values
  }


  unsafe fn set<T>(&mut self, buffer: &mut Self::Buffer, i: usize, x: T) -> Result<(), Self::Err> {
    self.state.borrow_mut().bind_array_buffer(*buffer);

    let ptr = gl::MapBuffer(gl::ARRAY_BUFFER, gl::WRITE_ONLY) as *mut T;
    *ptr.offset(i as isize) = x;
    let _ = gl::UnmapBuffer(gl::ARRAY_BUFFER);

    Ok(())
  }

  unsafe fn write_whole<T>(&self, buffer: &mut Self::Buffer, values: &[T], bytes: usize) -> Result<(), Self::Err> {
    self.state.borrow_mut().bind_array_buffer(*buffer);

    let ptr = gl::MapBuffer(gl::ARRAY_BUFFER, gl::WRITE_ONLY);
    ptr::copy_nonoverlapping(values.as_ptr() as *const c_void, ptr, bytes);
    let _ = gl::UnmapBuffer(gl::ARRAY_BUFFER);

    Ok(())
  }

  unsafe fn as_slice<T>(&mut self, buffer: &Self::Buffer) -> Result<*const T, Self::Err> {
    self.state.borrow_mut().bind_array_buffer(*buffer);

    let ptr = gl::MapBuffer(gl::ARRAY_BUFFER, gl::READ_ONLY) as *const T;

    if ptr.is_null() {
      return Err(BufferDriverError::MapFailed);
    }

    Ok(ptr)
  }

  unsafe fn as_slice_mut<T>(&mut self, buffer: &mut Self::Buffer) -> Result<*mut T, Self::Err> {
    self.state.borrow_mut().bind_array_buffer(*buffer);

    let ptr = gl::MapBuffer(gl::ARRAY_BUFFER, gl::READ_ONLY) as *mut T;

    if ptr.is_null() {
      return Err(BufferDriverError::MapFailed);
    }

    Ok(ptr)
  }

  unsafe fn drop_slice<T>(&mut self, buffer: &mut Self::Buffer, _: *const T) {
    self.state.borrow_mut().bind_array_buffer(*buffer);
    gl::UnmapBuffer(gl::ARRAY_BUFFER);
  }

  unsafe fn drop_slice_mut<T>(&mut self, buffer: &mut Self::Buffer, _: *mut T) {
    self.state.borrow_mut().bind_array_buffer(*buffer);
    gl::UnmapBuffer(gl::ARRAY_BUFFER);
  }
}

unsafe impl RenderStateDriver for GL33 {
  unsafe fn set_blending_state(&mut self, state: BlendingState) {
    match state {
      BlendingState::On => gl::Enable(gl::BLEND),
      BlendingState::Off => gl::Disable(gl::BLEND),
    }
  }

  unsafe fn set_blending_equation(&mut self, equation: Equation) {
    gl::BlendEquation(from_blending_equation(equation));
  }

  unsafe fn set_blending_func(&mut self, src: Factor, dest: Factor) {
    gl::BlendFunc(from_blending_factor(src), from_blending_factor(dest));
  }

  unsafe fn set_depth_test(&mut self, depth_test: DepthTest) {
    match depth_test {
      DepthTest::On => gl::Enable(gl::DEPTH_TEST),
      DepthTest::Off => gl::Disable(gl::DEPTH_TEST),
    }
  }

  unsafe fn set_face_culling_state(&mut self, state: FaceCullingState) {
    match state {
      FaceCullingState::On => gl::Enable(gl::CULL_FACE),
      FaceCullingState::Off => gl::Disable(gl::CULL_FACE),
    }
  }

  unsafe fn set_face_culling_order(&mut self, order: FaceCullingOrder) {
    match order {
      FaceCullingOrder::CW => gl::FrontFace(gl::CW),
      FaceCullingOrder::CCW => gl::FrontFace(gl::CCW),
    }
  }

  unsafe fn set_face_culling_mode(&mut self, mode: FaceCullingMode) {
    match mode {
      FaceCullingMode::Front => gl::CullFace(gl::FRONT),
      FaceCullingMode::Back => gl::CullFace(gl::BACK),
      FaceCullingMode::Both => gl::CullFace(gl::FRONT_AND_BACK),
    }
  }
}

#[inline]
fn from_blending_equation(equation: Equation) -> GLenum {
  match equation {
    Equation::Additive => gl::FUNC_ADD,
    Equation::Subtract => gl::FUNC_SUBTRACT,
    Equation::ReverseSubtract => gl::FUNC_REVERSE_SUBTRACT,
    Equation::Min => gl::MIN,
    Equation::Max => gl::MAX,
  }
}

#[inline]
fn from_blending_factor(factor: Factor) -> GLenum {
  match factor {
    Factor::One => gl::ONE,
    Factor::Zero => gl::ZERO,
    Factor::SrcColor => gl::SRC_COLOR,
    Factor::SrcColorComplement => gl::ONE_MINUS_SRC_COLOR,
    Factor::DestColor => gl::DST_COLOR,
    Factor::DestColorComplement => gl::ONE_MINUS_DST_COLOR,
    Factor::SrcAlpha => gl::SRC_ALPHA,
    Factor::SrcAlphaComplement => gl::ONE_MINUS_SRC_ALPHA,
    Factor::DstAlpha => gl::DST_ALPHA,
    Factor::DstAlphaComplement => gl::ONE_MINUS_DST_ALPHA,
    Factor::SrcAlphaSaturate => gl::SRC_ALPHA_SATURATE,
  }
}
