use gl;
use gl::types::*;

use std::fmt;

use crate::shader::stage2::Type;

#[derive(Debug)]
pub struct Stage {
  pub(crate) handle: GLuint
}

#[derive(Clone, Debug)]
pub enum StageError {
  /// Occurs when a shader fails to compile.
  CompilationFailed(Type, String),
  /// Occurs when you try to create a shader which type is not supported on the current hardware.
  UnsupportedType(Type),
}

impl fmt::Display for StageError {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      StageError::CompilationFailed(ref ty, ref r) => write!(f, "{} compilation error: {}", ty, r),
      StageError::UnsupportedType(ty) => write!(f, "unsupported {}", ty),
    }
  }
}

pub(crate) fn opengl_shader_type(t: Type) -> GLenum {
  match t {
    Type::TessellationControlShader => gl::TESS_CONTROL_SHADER,
    Type::TessellationEvaluationShader => gl::TESS_EVALUATION_SHADER,
    Type::VertexShader => gl::VERTEX_SHADER,
    Type::GeometryShader => gl::GEOMETRY_SHADER,
    Type::FragmentShader => gl::FRAGMENT_SHADER,
  }
}

pub(crate) fn glsl_pragma_src(src: &str) -> String {
  let mut pragma = String::from(GLSL_PRAGMA);
  pragma.push_str(src);
  pragma
}

pub(crate) const GLSL_PRAGMA: &'static str = "\
#version 330 core
#extension GL_ARB_separate_shader_objects : require
";
