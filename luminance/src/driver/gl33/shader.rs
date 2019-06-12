pub mod program;
pub mod stage;

use gl;
use gl::types::*;
use std::ffi::CString;
use std::fmt;
use std::ptr::{null, null_mut};

use crate::driver::ShaderDriver;
use crate::driver::gl33::GL33;
use crate::driver::gl33::shader::program::{Program, ProgramError, UniformBuilder, UniformWarning};
use crate::driver::gl33::shader::stage::{GLSL_PRAGMA, Stage, StageError, glsl_pragma_src, opengl_shader_type};
use crate::shader::stage2::Type;

#[derive(Debug)]
pub enum ShaderError {
  /// A shader stage error.
  StageError(StageError),
  /// A shader program error.
  ProgramError(ProgramError),
}

impl fmt::Display for ShaderError {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      ShaderError::StageError(ref e) => write!(f, "shader stage error: {}", e),
      ShaderError::ProgramError(ref e) => write!(f, "shader program error: {}", e),
    }
  }
}

impl From<StageError> for ShaderError {
  fn from(e: StageError) -> Self {
    ShaderError::StageError(e)
  }
}

impl From<ProgramError> for ShaderError {
  fn from(e: ProgramError) -> Self {
    ShaderError::ProgramError(e)
  }
}

pub struct Uniform {
  program: GLuint,
  index: GLint
}

impl Uniform {
  fn new(program: GLuint, index: GLint) -> Self {
    Uniform { program, index }
  }
}

unsafe impl ShaderDriver for GL33 {
  type Stage = Stage;

  type Program = Program;

  type UniformBuilder = UniformBuilder;

  type Uniform = Uniform;

  type Err = ShaderError;

  unsafe fn new_shader_stage(ty: Type, src: &str) -> Result<Self::Stage, Self::Err> {
    let handle = gl::CreateShader(opengl_shader_type(ty));

    if handle == 0 {
      return Err(
        StageError::CompilationFailed(
          ty,
          "unable to create shader stage".to_owned(),
        ).into()
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

      Err(StageError::CompilationFailed(ty, String::from_utf8(log).unwrap()).into())
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

  unsafe fn new_shader_program<'a, T, G>(
    tess: T,
    vertex: &'a Self::Stage,
    geometry: G,
    fragment: &'a Self::Stage
  ) -> Result<Self::Program, Self::Err>
  where T: Into<Option<(&'a Self::Stage, &'a Self::Stage)>>,
        G: Into<Option<&'a Self::Stage>> {
    let handle = gl::CreateProgram();

    if let Some((tcs, tes)) = tess.into() {
      gl::AttachShader(handle, tcs.handle);
      gl::AttachShader(handle, tes.handle);
    }

    gl::AttachShader(handle, vertex.handle);

    if let Some(geometry) = geometry.into() {
      gl::AttachShader(handle, geometry.handle);
    }

    gl::AttachShader(handle, fragment.handle);

    let program = Program { handle };
    Self::link_shader_program(&program).map(move |_| program)
  }

  unsafe fn drop_shader_program(program: &mut Self::Program) {
    gl::DeleteProgram(program.handle);
  }

  unsafe fn link_shader_program(program: &Self::Program) -> Result<(), Self::Err> {
    let handle = program.handle;

    gl::LinkProgram(handle);

    let mut linked: GLint = gl::FALSE as GLint;
    gl::GetProgramiv(handle, gl::LINK_STATUS, &mut linked);

    if linked == (gl::TRUE as GLint) {
      Ok(())
    } else {
      let mut log_len: GLint = 0;
      gl::GetProgramiv(handle, gl::INFO_LOG_LENGTH, &mut log_len);

      let mut log: Vec<u8> = Vec::with_capacity(log_len as usize);
      gl::GetProgramInfoLog(handle, log_len, null_mut(), log.as_mut_ptr() as *mut GLchar);

      gl::DeleteProgram(handle);

      log.set_len(log_len as usize);

      Err(ProgramError::LinkFailed(String::from_utf8(log).unwrap()).into())
    }
  }

  unsafe fn new_uniform_builder(
    _: &mut Self::Program
  ) -> Result<Self::UniformBuilder, Self::Err> {
    Ok(UniformBuilder::default())
  }

  unsafe fn ask_uniform(
    program: &mut Self::Program,
    _: &mut Self::UniformBuilder,
    name: &str
  ) -> Result<Self::Uniform, Self::Err> {
    let c_name = CString::new(name.as_bytes()).unwrap();
    let location = gl::GetUniformLocation(program.handle, c_name.as_ptr() as *const GLchar);

    if location < 0 {
      Err(ProgramError::UniformWarning(UniformWarning::Inactive(name.to_owned())).into())
    } else {
      Ok(Uniform::new(program.handle, location))
    }
  }

  unsafe fn ask_uniform_block(
    program: &mut Self::Program,
    _: &mut Self::UniformBuilder,
    name: &str
  ) -> Result<Self::Uniform, Self::Err> {
    let c_name = CString::new(name.as_bytes()).unwrap();
    let location = gl::GetUniformBlockIndex(program.handle, c_name.as_ptr() as *const GLchar);

    if location == gl::INVALID_INDEX {
      Err(ProgramError::UniformWarning(UniformWarning::Inactive(name.to_owned())).into())
    } else {
      Ok(Uniform::new(program.handle, location as GLint))
    }
  }

  unsafe fn ask_unbound_uniform(
    program: &mut Self::Program,
    _: &mut Self::UniformBuilder
  ) -> Result<Self::Uniform, Self::Err> {
    Ok(Uniform::new(program.handle, -1))
  }
}
