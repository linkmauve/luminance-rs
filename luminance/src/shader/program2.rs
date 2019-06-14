//! Shader programs related types and functions.
//!
//! A shader `Program` is an object representing several operations. It’s a streaming program that
//! will operate on vertices, vertex patches, primitives and/or fragments.
//!
//! > *Note: shader programs don’t have to run on all those objects; they can be ran only on
//! vertices and fragments, for instance*.
//!
//! Creating a shader program is very simple. You need shader `Stage`s representing each step of the
//! processing.
//!
//! You *have* to provide at least a vertex and a fragment stages. If you want tessellation
//! processing, you need to provide a tessellation control and tessellation evaluation stages. If
//! you want primitives processing, you need to add a geometry stage.
//!
//! In order to customize the behavior of your shader programs, you have access to *uniforms*. For
//! more details about them, see the documentation for the type `Uniform` and trait `Uniformable`.
//! When creating a new shader program, you have to provide code to declare its *uniform semantics*.
//!
//! The *uniform semantics* represent a mapping between the variables declared in your shader
//! sources and variables you have access in your host code in Rust. Typically, you declare your
//! variable – `Uniform` – in Rust as `const` and use the function `Uniform::sem` to get the
//! semantic associated with the string you pass in.
//!
//! > **Becareful: currently, uniforms are a bit messy as you have to provide a per-program unique
//! number when you use the `Uniform::new` method. Efforts will be done in that direction in later
//! releases.
//!
//! You can create a `Program` with its `new` associated function.

use std::fmt;
use std::marker::PhantomData;
use std::ops::Deref;

use crate::driver::{ShaderDriver, UniformDriver};
use crate::linear::{M22, M33, M44};
use crate::shader::stage2::{self, Stage, StageError};
use crate::vertex::Semantics;

/// A typed shader program.
///
/// Typed shader programs represent their inputs, outputs and environment (uniforms) directly in
/// their types. This is very interesting as it adds more static safety and enables such programs
/// to *“store”* information like the uniform interface and such.
pub struct Program<D, S, Out, Uni> where D: ShaderDriver {
  inner: D::Program,
  uni_iface: Uni,
  _in: PhantomData<*const S>,
  _out: PhantomData<*const Out>,
}

impl<D, S, Out, Uni> Program<D, S, Out, Uni>
where D: ShaderDriver,
      S: Semantics {
  /// Create a new program by consuming `Stage`s.
  pub fn from_stages<'a, T, G>(
    tess: T,
    vertex: &Stage<D>,
    geometry: G,
    fragment: &Stage<D>,
  ) -> Result<(Self, Vec<ProgramWarning>), D::Err>
  where Uni: UniformInterface,
        T: Into<Option<(&'a Stage<D>, &'a Stage<D>)>>,
        G: Into<Option<&'a Stage<D>>> {
    Self::from_stages_env(tess, vertex, geometry, fragment, ())
  }

  /// Create a new program by consuming strings.
  pub fn from_strings<'a, T, G>(
    tess: T,
    vertex: &str,
    geometry: G,
    fragment: &str,
  ) -> Result<(Self, Vec<ProgramWarning>), D::Err>
  where Uni: UniformInterface,
        T: Into<Option<(&'a str, &'a str)>>,
        G: Into<Option<&'a str>> {
    Self::from_strings_env(tess, vertex, geometry, fragment, ())
  }

  /// Create a new program by consuming `Stage`s and by looking up an environment.
  pub fn from_stages_env<'a, E, T, G>(
    tess: T,
    vertex: &Stage<D>,
    geometry: G,
    fragment: &Stage<D>,
    env: E,
  ) -> Result<(Self, Vec<ProgramWarning>), D::Err>
  where Uni: UniformInterface<E>,
        T: Into<Option<(&'a Stage<D>, &'a Stage<D>)>>,
        G: Into<Option<&'a Stage<D>>> {
    let raw = RawProgram::new(tess, vertex, geometry, fragment)?;
    let program = unsafe { D::new_shader_program(tess, vertex, geometry, fragment)? };

    let mut warnings = bind_vertex_attribs_locations::<S>(&raw);

    raw.link()?;

    let (uni_iface, uniform_warnings) = create_uniform_interface(&raw, env)?;
    warnings.extend(uniform_warnings.into_iter().map(ProgramWarning::Uniform));

    let program = Program {
      raw,
      uni_iface,
      _in: PhantomData,
      _out: PhantomData,
    };

    Ok((program, warnings))
  }

  /// Create a new program by consuming strings.
  pub fn from_strings_env<'a, E, T, G>(
    tess: T,
    vertex: &str,
    geometry: G,
    fragment: &str,
    env: E,
  ) -> Result<(Self, Vec<ProgramWarning>), D::Err>
  where Uni: UniformInterface<E>,
        T: Into<Option<(&'a str, &'a str)>>,
        G: Into<Option<&'a str>> {
    let tess = match tess.into() {
      Some((tcs_str, tes_str)) => {
        let tcs =
          Stage::new(stage::Type::TessellationControlShader, tcs_str).map_err(ProgramError::StageError)?;
        let tes =
          Stage::new(stage::Type::TessellationControlShader, tes_str).map_err(ProgramError::StageError)?;
        Some((tcs, tes))
      }
      None => None,
    };

    let gs = match geometry.into() {
      Some(gs_str) => {
        Some(Stage::new(stage::Type::GeometryShader, gs_str).map_err(ProgramError::StageError)?)
      }
      None => None,
    };

    let vs = Stage::new(stage::Type::VertexShader, vertex).map_err(ProgramError::StageError)?;
    let fs = Stage::new(stage::Type::FragmentShader, fragment).map_err(ProgramError::StageError)?;

    Self::from_stages_env(
      tess.as_ref().map(|&(ref tcs, ref tes)| (tcs, tes)),
      &vs,
      gs.as_ref(),
      &fs,
      env,
    )
  }

  // /// Get the program interface associated with this program.
  // pub(crate) fn interface<'a>(&'a self) -> ProgramInterface<'a, Uni> {
  //   let raw_program = &self.raw;
  //   let uniform_interface = &self.uni_iface;

  //   ProgramInterface {
  //     raw_program,
  //     uniform_interface,
  //   }
  // }

  // /// Transform the program to adapt the uniform interface.
  // ///
  // /// This function will not re-allocate nor recreate the GPU data. It will try to change the
  // /// uniform interface and if the new uniform interface is correctly generated, return the same
  // /// shader program updated with the new uniform interface. If the generation of the new uniform
  // /// interface fails, this function will return the program with the former uniform interface.
  // pub fn adapt<Q>(self) -> Result<(Program<S, Out, Q>, Vec<UniformWarning>), (ProgramError, Self)>
  // where Q: UniformInterface {
  //   self.adapt_env(())
  // }

  // /// Transform the program to adapt the uniform interface by looking up an environment.
  // ///
  // /// This function will not re-allocate nor recreate the GPU data. It will try to change the
  // /// uniform interface and if the new uniform interface is correctly generated, return the same
  // /// shader program updated with the new uniform interface. If the generation of the new uniform
  // /// interface fails, this function will return the program with the former uniform interface.
  // pub fn adapt_env<Q, E>(
  //   self,
  //   env: E,
  // ) -> Result<(Program<S, Out, Q>, Vec<UniformWarning>), (ProgramError, Self)>
  // where Q: UniformInterface<E> {
  //   // first, try to create the new uniform interface
  //   let new_uni_iface = create_uniform_interface(&self.raw, env);

  //   match new_uni_iface {
  //     Ok((uni_iface, warnings)) => {
  //       // if we have succeeded, return self with the new uniform interface
  //       let program = Program {
  //         raw: self.raw,
  //         uni_iface,
  //         _in: PhantomData,
  //         _out: PhantomData,
  //       };

  //       Ok((program, warnings))
  //     }

  //     Err(iface_err) => {
  //       // we couldn’t generate the new uniform interface; return the error(s) that occurred and the
  //       // the untouched former program
  //       Err((iface_err, self))
  //     }
  //   }
  // }

  // /// A version of [`Program::adapt_env`] that doesn’t change the uniform interface type.
  // ///
  // /// This function might be needed for when you want to update the uniform interface but still
  // /// enforce that the type must remain the same.
  // pub fn readapt_env<E>(self, env: E) -> Result<(Self, Vec<UniformWarning>), (ProgramError, Self)>
  // where Uni: UniformInterface<E> {
  //   self.adapt_env(env)
  // }
}

/// Class of types that can act as uniform interfaces in typed programs.
///
/// A uniform interface is a value that contains uniforms. The purpose of a uniform interface is to
/// be stored in a typed program and handed back when the program is made available in a pipeline.
///
/// The `E` type variable represents the environment and might be used to drive the implementation
/// from a value. It’s defaulted to `()` so that if you don’t use the environment, you don’t have to
/// worry about that value when creating the shader program.
pub trait UniformInterface<E = ()>: Sized {
  /// Build the uniform interface.
  ///
  /// When mapping a uniform, if you want to accept failures, you can discard the error and use
  /// `UniformBuilder::unbound` to let the uniform pass through, and collect the uniform warning.
  fn uniform_interface<'a, D>(
    builder: &mut UniformBuilder<'a, D>,
    env: E
  ) -> Result<Self, ProgramError>
  where D: ShaderDriver;
}

impl UniformInterface for () {
  fn uniform_interface<'a, D>(
    _: &mut UniformBuilder<'a, D>,
    _: E
  ) -> Result<Self, ProgramError>
  where D: ShaderDriver {
    Ok(())
  }
}

/// Build uniforms to fold them to a uniform interface.
pub struct UniformBuilder<'a, D> where D: ShaderDriver {
  program: &'a D::Program,
  inner: D::UniformBuilder,
}

impl<'a, D> UniformBuilder<'a, D> where D: ShaderDriver {
  fn new(program: &'a D::Program) -> Self {
    let inner = unsafe { D::new_uniform_builder(program) };

    UniformBuilder {
      program,
      inner,
    }
  }

  /// Have the builder hand you a `Uniform` of the type of your choice.
  ///
  /// Keep in mind that it’s possible that this function fails if you ask for a type for which the
  /// one defined in the shader doesn’t type match. If you don’t want a failure but an *unbound*
  /// uniform, head over to the `ask_unbound` function.
  pub fn ask<T>(&self, name: &str) -> Result<Uniform<D, T>, D::Err>
  where T: Uniformable {
    let uniform = match T::ty() {
      Type::BufferBinding => self.ask_uniform_block(name)?,
      _ => self.ask_uniform(name)?,
    };

    uniform_type_match(self.raw.handle, name, T::ty())
      .map_err(|err| UniformWarning::TypeMismatch(name.to_owned(), err))?;

    Ok(uniform)
  }

  pub fn ask_unbound<T>(&mut self, name: &str) -> Uniform<D, T>
  where T: Uniformable {
    match self.ask(name) {
      Ok(uniform) => uniform,
      Err(warning) => {
        self.warnings.push(warning);
        self.unbound()
      }
    }
  }

  fn ask_uniform<T>(&self, name: &str) -> Result<Uniform<D, T>, D::Err>
  where T: Uniformable {
    unsafe {
      D::ask_uniform(&self.program.inner, &self.inner).map(Uniform::new)
    }
  }

  fn ask_uniform_block<T>(&self, name: &str) -> Result<Uniform<D, T>, UniformWarning>
  where T: Uniformable {
    unsafe {
      D::ask_uniform_block(&self.program.inner, &self.inner).map(Uniform::new)
    }
  }

  /// Special uniform that won’t do anything.
  ///
  /// Use that function when you need a uniform to complete a uniform interface but you’re sure you
  /// won’t use it.
  pub fn unbound<T>(&self) -> Uniform<D, T> where T: Uniformable {
    unsafe { D::unbound_uniform(&self.program.inner, &self.inner).map(Uniform::new) }
  }
}

// /// The shader program interface.
// ///
// /// This struct gives you access to several capabilities, among them:
// ///
// ///   - The typed *uniform interface* you would have acquired earlier.
// ///   - Some functions to query more data dynamically.
// pub struct ProgramInterface<'a, Uni> {
//   raw_program: &'a RawProgram,
//   uniform_interface: &'a Uni,
// }
//
// impl<'a, Uni> Deref for ProgramInterface<'a, Uni> {
//   type Target = Uni;
//
//   fn deref(&self) -> &Self::Target {
//     self.uniform_interface
//   }
// }
//
// impl<'a, Uni> ProgramInterface<'a, Uni> {
//   pub fn query(&'a self) -> UniformBuilder<'a> {
//     UniformBuilder::new(self.raw_program)
//   }
// }

/// A contravariant shader uniform. `Uniform<T>` doesn’t hold any value. It’s more like a mapping
/// between the host code and the shader the uniform was retrieved from.
#[derive(Debug)]
pub struct Uniform<D, T> where D: ShaderDriver {
  inner: D::Uniform,
  _D: PhantomData<*const D>,
  _t: PhantomData<*const T>,
}

// impl<D, T> Uniform<D, T>
// where D: UniformDriver<T>,
//       T: Uniformable {
//   fn new(program: GLuint, index: GLint) -> Self {
//     Uniform {
//       program,
//       index,
//       _d: PhantomData,
//       _t: PhantomData,
//     }
//   }
//
//   fn unbound(program: GLuint) -> Self {
//     Uniform {
//       program,
//       index: -1,
//       _t: PhantomData,
//     }
//   }
//
//   pub(crate) fn program(&self) -> GLuint {
//     self.program
//   }
//
//   pub(crate) fn index(&self) -> GLint {
//     self.index
//   }
//
//   /// Update the value pointed by this uniform.
//   pub fn update(&self, x: T) {
//     x.update(self);
//   }
// }

/// Type of a uniform.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Type {
  // scalars
  Int,
  UInt,
  Float,
  Bool,
  // vectors
  IVec2,
  IVec3,
  IVec4,
  UIVec2,
  UIVec3,
  UIVec4,
  Vec2,
  Vec3,
  Vec4,
  BVec2,
  BVec3,
  BVec4,
  // matrices
  M22,
  M33,
  M44,
  // textures
  ISampler1D,
  ISampler2D,
  ISampler3D,
  UISampler1D,
  UISampler2D,
  UISampler3D,
  Sampler1D,
  Sampler2D,
  Sampler3D,
  ICubemap,
  UICubemap,
  Cubemap,
  // buffer
  BufferBinding,
}

/// Types that can behave as `Uniform`.
pub unsafe trait Uniformable: Sized {
  /// Retrieve the [`Type`] of the uniform.
  fn ty() -> Type;
}

//unsafe impl Uniformable for i32 {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe { gl::Uniform1i(u.index, self) }
//  }
//
//  fn ty() -> Type {
//    Type::Int
//  }
//}
//
//unsafe impl Uniformable for [i32; 2] {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe { gl::Uniform2iv(u.index, 1, &self as *const i32) }
//  }
//
//  fn ty() -> Type {
//    Type::IVec2
//  }
//}
//
//unsafe impl Uniformable for [i32; 3] {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe { gl::Uniform3iv(u.index, 1, &self as *const i32) }
//  }
//
//  fn ty() -> Type {
//    Type::IVec3
//  }
//}
//
//unsafe impl Uniformable for [i32; 4] {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe { gl::Uniform4iv(u.index, 1, &self as *const i32) }
//  }
//
//  fn ty() -> Type {
//    Type::IVec4
//  }
//}
//
//unsafe impl<'a> Uniformable for &'a [i32] {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe { gl::Uniform1iv(u.index, self.len() as GLsizei, self.as_ptr()) }
//  }
//
//  fn ty() -> Type {
//    Type::Int
//  }
//}
//
//unsafe impl<'a> Uniformable for &'a [[i32; 2]] {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe { gl::Uniform2iv(u.index, self.len() as GLsizei, self.as_ptr() as *const i32) }
//  }
//
//  fn ty() -> Type {
//    Type::IVec2
//  }
//}
//
//unsafe impl<'a> Uniformable for &'a [[i32; 3]] {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe { gl::Uniform3iv(u.index, self.len() as GLsizei, self.as_ptr() as *const i32) }
//  }
//
//  fn ty() -> Type {
//    Type::IVec3
//  }
//}
//
//unsafe impl<'a> Uniformable for &'a [[i32; 4]] {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe { gl::Uniform4iv(u.index, self.len() as GLsizei, self.as_ptr() as *const i32) }
//  }
//
//  fn ty() -> Type {
//    Type::IVec4
//  }
//}
//
//unsafe impl Uniformable for u32 {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe { gl::Uniform1ui(u.index, self) }
//  }
//
//  fn ty() -> Type {
//    Type::UInt
//  }
//}
//
//unsafe impl Uniformable for [u32; 2] {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe { gl::Uniform2uiv(u.index, 1, &self as *const u32) }
//  }
//
//  fn ty() -> Type {
//    Type::UIVec2
//  }
//}
//
//unsafe impl Uniformable for [u32; 3] {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe { gl::Uniform3uiv(u.index, 1, &self as *const u32) }
//  }
//
//  fn ty() -> Type {
//    Type::UIVec3
//  }
//}
//
//unsafe impl Uniformable for [u32; 4] {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe { gl::Uniform4uiv(u.index, 1, &self as *const u32) }
//  }
//
//  fn ty() -> Type {
//    Type::UIVec4
//  }
//}
//
//unsafe impl<'a> Uniformable for &'a [u32] {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe { gl::Uniform1uiv(u.index, self.len() as GLsizei, self.as_ptr() as *const u32) }
//  }
//
//  fn ty() -> Type {
//    Type::UInt
//  }
//}
//
//unsafe impl<'a> Uniformable for &'a [[u32; 2]] {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe { gl::Uniform2uiv(u.index, self.len() as GLsizei, self.as_ptr() as *const u32) }
//  }
//
//  fn ty() -> Type {
//    Type::UIVec2
//  }
//}
//
//unsafe impl<'a> Uniformable for &'a [[u32; 3]] {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe { gl::Uniform3uiv(u.index, self.len() as GLsizei, self.as_ptr() as *const u32) }
//  }
//
//  fn ty() -> Type {
//    Type::UIVec3
//  }
//}
//
//unsafe impl<'a> Uniformable for &'a [[u32; 4]] {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe { gl::Uniform4uiv(u.index, self.len() as GLsizei, self.as_ptr() as *const u32) }
//  }
//
//  fn ty() -> Type {
//    Type::UIVec4
//  }
//}
//
//unsafe impl Uniformable for f32 {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe { gl::Uniform1f(u.index, self) }
//  }
//
//  fn ty() -> Type {
//    Type::Float
//  }
//}
//
//unsafe impl Uniformable for [f32; 2] {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe { gl::Uniform2fv(u.index, 1, &self as *const f32) }
//  }
//
//  fn ty() -> Type {
//    Type::Vec2
//  }
//}
//
//unsafe impl Uniformable for [f32; 3] {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe { gl::Uniform3fv(u.index, 1, &self as *const f32) }
//  }
//
//  fn ty() -> Type {
//    Type::Vec3
//  }
//}
//
//unsafe impl Uniformable for [f32; 4] {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe { gl::Uniform4fv(u.index, 1, &self as *const f32) }
//  }
//
//  fn ty() -> Type {
//    Type::Vec4
//  }
//}
//
//unsafe impl<'a> Uniformable for &'a [f32] {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe { gl::Uniform1fv(u.index, self.len() as GLsizei, self.as_ptr() as *const f32) }
//  }
//
//  fn ty() -> Type {
//    Type::Float
//  }
//}
//
//unsafe impl<'a> Uniformable for &'a [[f32; 2]] {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe { gl::Uniform2fv(u.index, self.len() as GLsizei, self.as_ptr() as *const f32) }
//  }
//
//  fn ty() -> Type {
//    Type::Vec2
//  }
//}
//
//unsafe impl<'a> Uniformable for &'a [[f32; 3]] {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe { gl::Uniform3fv(u.index, self.len() as GLsizei, self.as_ptr() as *const f32) }
//  }
//
//  fn ty() -> Type {
//    Type::Vec3
//  }
//}
//
//unsafe impl<'a> Uniformable for &'a [[f32; 4]] {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe { gl::Uniform4fv(u.index, self.len() as GLsizei, self.as_ptr() as *const f32) }
//  }
//
//  fn ty() -> Type {
//    Type::Vec4
//  }
//}
//
//unsafe impl Uniformable for M22 {
//  fn update(self, u: &Uniform<Self>) {
//    let v = [self];
//    unsafe { gl::UniformMatrix2fv(u.index, 1, gl::FALSE, v.as_ptr() as *const f32) }
//  }
//
//  fn ty() -> Type {
//    Type::M22
//  }
//}
//
//unsafe impl Uniformable for M33 {
//  fn update(self, u: &Uniform<Self>) {
//    let v = [self];
//    unsafe { gl::UniformMatrix3fv(u.index, 1, gl::FALSE, v.as_ptr() as *const f32) }
//  }
//
//  fn ty() -> Type {
//    Type::M33
//  }
//}
//
//unsafe impl Uniformable for M44 {
//  fn update(self, u: &Uniform<Self>) {
//    let v = [self];
//    unsafe { gl::UniformMatrix4fv(u.index, 1, gl::FALSE, v.as_ptr() as *const f32) }
//  }
//
//  fn ty() -> Type {
//    Type::M44
//  }
//}
//
//unsafe impl<'a> Uniformable for &'a [M22] {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe {
//      gl::UniformMatrix2fv(
//        u.index,
//        self.len() as GLsizei,
//        gl::FALSE,
//        self.as_ptr() as *const f32,
//      )
//    }
//  }
//
//  fn ty() -> Type {
//    Type::M22
//  }
//}
//
//unsafe impl<'a> Uniformable for &'a [M33] {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe {
//      gl::UniformMatrix3fv(
//        u.index,
//        self.len() as GLsizei,
//        gl::FALSE,
//        self.as_ptr() as *const f32,
//      )
//    }
//  }
//
//  fn ty() -> Type {
//    Type::M33
//  }
//}
//
//unsafe impl<'a> Uniformable for &'a [M44] {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe {
//      gl::UniformMatrix4fv(
//        u.index,
//        self.len() as GLsizei,
//        gl::FALSE,
//        self.as_ptr() as *const f32,
//      )
//    }
//  }
//
//  fn ty() -> Type {
//    Type::M44
//  }
//}
//
//unsafe impl Uniformable for bool {
//  fn update(self, u: &Uniform<Self>) {
//    unsafe { gl::Uniform1ui(u.index, self as GLuint) }
//  }
//
//  fn ty() -> Type {
//    Type::Bool
//  }
//}
//
//unsafe impl Uniformable for [bool; 2] {
//  fn update(self, u: &Uniform<Self>) {
//    let v = [self[0] as u32, self[1] as u32];
//    unsafe { gl::Uniform2uiv(u.index, 1, &v as *const u32) }
//  }
//
//  fn ty() -> Type {
//    Type::BVec2
//  }
//}
//
//unsafe impl Uniformable for [bool; 3] {
//  fn update(self, u: &Uniform<Self>) {
//    let v = [self[0] as u32, self[1] as u32, self[2] as u32];
//    unsafe { gl::Uniform3uiv(u.index, 1, &v as *const u32) }
//  }
//
//  fn ty() -> Type {
//    Type::BVec3
//  }
//}
//
//unsafe impl Uniformable for [bool; 4] {
//  fn update(self, u: &Uniform<Self>) {
//    let v = [self[0] as u32, self[1] as u32, self[2] as u32, self[3] as u32];
//    unsafe { gl::Uniform4uiv(u.index, 1, &v as *const u32) }
//  }
//
//  fn ty() -> Type {
//    Type::BVec4
//  }
//}
//
//unsafe impl<'a> Uniformable for &'a [bool] {
//  fn update(self, u: &Uniform<Self>) {
//    let v: Vec<_> = self.iter().map(|x| *x as u32).collect();
//    unsafe { gl::Uniform1uiv(u.index, v.len() as GLsizei, v.as_ptr()) }
//  }
//
//  fn ty() -> Type {
//    Type::Bool
//  }
//}
//
//unsafe impl<'a> Uniformable for &'a [[bool; 2]] {
//  fn update(self, u: &Uniform<Self>) {
//    let v: Vec<_> = self.iter().map(|x| [x[0] as u32, x[1] as u32]).collect();
//    unsafe { gl::Uniform2uiv(u.index, v.len() as GLsizei, v.as_ptr() as *const u32) }
//  }
//
//  fn ty() -> Type {
//    Type::BVec2
//  }
//}
//
//unsafe impl<'a> Uniformable for &'a [[bool; 3]] {
//  fn update(self, u: &Uniform<Self>) {
//    let v: Vec<_> = self
//      .iter()
//      .map(|x| [x[0] as u32, x[1] as u32, x[2] as u32])
//      .collect();
//    unsafe { gl::Uniform3uiv(u.index, v.len() as GLsizei, v.as_ptr() as *const u32) }
//  }
//
//  fn ty() -> Type {
//    Type::BVec3
//  }
//}
//
//unsafe impl<'a> Uniformable for &'a [[bool; 4]] {
//  fn update(self, u: &Uniform<Self>) {
//    let v: Vec<_> = self
//      .iter()
//      .map(|x| [x[0] as u32, x[1] as u32, x[2] as u32, x[3] as u32])
//      .collect();
//    unsafe { gl::Uniform4uiv(u.index, v.len() as GLsizei, v.as_ptr() as *const u32) }
//  }
//
//  fn ty() -> Type {
//    Type::BVec4
//  }
//}
