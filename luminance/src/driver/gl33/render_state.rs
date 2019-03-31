use crate::blending::{BlendingState, Equation, Factor};
use crate::depth_test::DepthTest;
use crate::driver::RenderStateDriver;
use crate::driver::gl33::GL33;
use crate::face_culling::{FaceCullingMode, FaceCullingOrder, FaceCullingState};
use gl;
use gl::types::*;

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

