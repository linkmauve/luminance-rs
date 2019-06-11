use std::ffi::CString;
use std::fmt;
use std::ptr::{null, null_mut};

use crate::driver::ShaderDriver;

/// A shader stage type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Type {
  TessellationControlShader,
  TessellationEvaluationShader,
  VertexShader,
  GeometryShader,
  FragmentShader,
}

impl fmt::Display for Type {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      Type::TessellationControlShader => f.write_str("tessellation control shader"),
      Type::TessellationEvaluationShader => f.write_str("tessellation evaluation shader"),
      Type::VertexShader => f.write_str("vertex shader"),
      Type::GeometryShader => f.write_str("geometry shader"),
      Type::FragmentShader => f.write_str("fragment shader"),
    }
  }
}

/// A shader stage.
#[derive(Debug)]
pub struct Stage<D> where D: ?Sized + ShaderDriver {
  inner: D::Stage,
  ty: Type,
}

impl<D> Stage<D> where D: ?Sized + ShaderDriver {
  /// Create a new shader stage.
  pub fn new<S>(ty: Type, src: S) -> Result<Self, StageError<D>> where S: AsRef<str> {
    let src = src.as_ref();

    unsafe {
      D::new_shader_stage(ty, src)
        .map(|inner| Stage { inner, ty })
        .map_err(StageError::DriverError)
    }
  }
}

impl<D> Drop for Stage<D> where D: ?Sized + ShaderDriver {
  fn drop(&mut self) {
    unsafe { D::drop_shader_stage(&mut self.inner) };
  }
}

/// Errors that shader stages can emit.
#[derive(Clone, Debug)]
pub enum StageError<D> where D: ?Sized + ShaderDriver {
  DriverError(D::Err),
}

impl<D> fmt::Display for StageError<D> where D: ?Sized + ShaderDriver {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      StageError::DriverError(ref e) => write!(f, "shader driver error: {}", e),
    }
  }
}
