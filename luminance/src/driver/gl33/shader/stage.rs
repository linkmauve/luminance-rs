use gl;
use gl::types::*;

use std::ffi::CString;
use std::fmt;
use std::ptr::{null, null_mut};

use crate::driver::ShaderDriver;
use crate::driver::gl33::GL33;
use crate::shader::stage2::Type;

pub struct Stage {
  handle: GLuint
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

unsafe impl ShaderDriver for GL33 {
  type Stage = Stage;

  type Err = StageError;

  unsafe fn new_shader_stage(ty: Type, src: &str) -> Result<Self::Stage, Self::Err> {
    let handle = gl::CreateShader(opengl_shader_type(ty));

    if handle == 0 {
      return Err(
        StageError::CompilationFailed(
          ty,
          "unable to create shader stage".to_owned(),
        )
      );
    }

    let mut stage = Stage { handle };

    Self::source_shader_stage(&mut stage, src);
    gl::CompileShader(handle);

    let mut compiled: GLint = gl::FALSE as GLint;
    gl::GetShaderiv(handle, gl::COMPILE_STATUS, &mut compiled);

    if compiled == (gl::TRUE as GLint) {
      Ok(stage)
    } else {
      let mut log_len: GLint = 0;
      gl::GetShaderiv(handle, gl::INFO_LOG_LENGTH, &mut log_len);

      let mut log: Vec<u8> = Vec::with_capacity(log_len as usize);
      gl::GetShaderInfoLog(handle, log_len, null_mut(), log.as_mut_ptr() as *mut GLchar);

      gl::DeleteShader(handle);

      log.set_len(log_len as usize);

      Err(StageError::CompilationFailed(ty, String::from_utf8(log).unwrap()))
    }
  }

  unsafe fn drop_shader_stage(stage: &mut Self::Stage) {
    gl::DeleteShader(stage.handle);
  }

  unsafe fn source_shader_stage(stage: &mut Self::Stage, src: &str) -> Result<(), Self::Err> {
    let c_src = CString::new(glsl_pragma_src(src).as_bytes()).unwrap();
    gl::ShaderSource(stage.handle, 1, [c_src.as_ptr()].as_ptr(), null());
    Ok(())
  }
}

fn opengl_shader_type(t: Type) -> GLenum {
  match t {
    Type::TessellationControlShader => gl::TESS_CONTROL_SHADER,
    Type::TessellationEvaluationShader => gl::TESS_EVALUATION_SHADER,
    Type::VertexShader => gl::VERTEX_SHADER,
    Type::GeometryShader => gl::GEOMETRY_SHADER,
    Type::FragmentShader => gl::FRAGMENT_SHADER,
  }
}

fn glsl_pragma_src(src: &str) -> String {
  let mut pragma = String::from(GLSL_PRAGMA);
  pragma.push_str(src);
  pragma
}

const GLSL_PRAGMA: &'static str = "\
#version 330 core
#extension GL_ARB_separate_shader_objects : require
";
