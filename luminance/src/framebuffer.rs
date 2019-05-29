//! Framebuffers and utility types and functions.
//!
//! Framebuffers are at the core of rendering. They’re the support of rendering operations and can
//! be used to highly enhance the visual aspect of a render. You’re always provided with at least
//! one framebuffer, `Framebuffer::back_buffer`. That function returns a framebuffer that represents –
//! for short – the current back framebuffer. You can render to that framebuffer and when you
//! *swap* the buffers, your render appears in the front framebuffer (likely your screen).
//!
//! # Framebuffers
//!
//! A framebuffer is an object maintaining the required GPU state to hold images you render to. It
//! gathers two important concepts:
//!
//!   - *Color buffers*.
//!   - *Depth buffers*.
//!
//! The *color buffers* hold the color images you render to. A framebuffer can hold several of them
//! with different color formats. The *depth buffers* hold the depth images you render to.
//! Framebuffers can hold only one depth buffer.
//!
//! # Framebuffer slots
//!
//! A framebuffer slot contains either its color buffers or its depth buffer. Sometimes, you might
//! find it handy to have no slot at all for a given type of buffer. In that case, we use `()`.
//!
//! The slots are a way to convert the different formats you use for your framebuffers’ buffers into
//! their respective texture representation so that you can handle the corresponding texels.
//!
//! Color buffers are abstracted by `ColorSlot` and the depth buffer by `DepthSlot`.

use std::fmt;
use std::marker::PhantomData;

use crate::context::GraphicsContext;
use crate::driver::FramebufferDriver;
use crate::pixel::{ColorPixel, DepthPixel, PixelFormat, RenderablePixel};
use crate::texture::{
  opengl_target, Dim2, Dimensionable, Flat, Layerable, RawTexture, Texture, TextureError,
};

/// Framebuffer with static layering, dimension, access and slots formats.
///
/// A `Framebuffer` is a *GPU* special object used to render to. Because framebuffers have a
/// *layering* property, it’s possible to have regular render and *layered rendering*. The dimension
/// of a framebuffer makes it possible to render to 1D, 2D, 3D and cubemaps.
///
/// A framebuffer has two kind of slots:
///
/// - **color slot** ;
/// - **depth slot**.
///
/// A framebuffer can have zero or several color slots and it can have zero or one depth slot. If
/// you use several color slots, you’ll be performing what’s called *MRT* (*M* ultiple *R* ender
/// *T* argets), enabling to render to several textures at once.
#[derive(Debug)]
pub struct Framebuffer<X, L, D, CS, DS>
where X: FramebufferDriver,
      L: Layerable,
      D: Dimensionable,
      D::Size: Copy,
      CS: ColorSlot<X, L, D>,
      DS: DepthSlot<X, L, D>, {
  raw: X::Framebuffer,
  w: u32,
  h: u32,
  color_slot: CS::ColorTextures,
  depth_slot: DS::DepthTexture,
  _l: PhantomData<L>,
  _d: PhantomData<D>,
}

impl<X, L, D, CS, DS> Drop for Framebuffer<X, L, D, CS, DS>
where X: FramebufferDriver,
      L: Layerable,
      D: Dimensionable,
      D::Size: Copy,
      CS: ColorSlot<X, L, D>,
      DS: DepthSlot<X, L, D>, {
  fn drop(&mut self) {
    X::drop_framebuffer(&mut self.raw)
  }
}

impl<X, L, D, CS, DS> Framebuffer<L, D, CS, DS>
where X: FramebufferDriver,
      L: Layerable,
      D: Dimensionable,
      D::Size: Copy,
      CS: ColorSlot<X, L, D>,
      DS: DepthSlot<X, L, D>, {
  /// Create a new farmebuffer.
  ///
  /// You’re always handed at least the base level of the texture. If you require any *additional*
  /// levels, you can pass the number via the `mipmaps` parameter.
  pub fn new<C>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
  ) -> Result<Framebuffer<L, D, CS, DS>, <X as FramebufferDriver>::Err>
  where C: GraphicsContext<Driver = X> {
    let mipmaps = mipmaps + 1;
    let (raw, cs, ds) = X::new_framebuffer::<L, D, CS, DS>(size, mipmaps)?;

    Ok(Framebuffer {
      raw,
      w: D::width(size),
      h: D::height(size),
      color_slot: cs,
      depth_slot: ds,
      _l: PhantomData,
      _d: PhantomData,
    })
  }

  #[inline]
  pub fn width(&self) -> u32 {
    self.w
  }

  #[inline]
  pub fn height(&self) -> u32 {
    self.h
  }

  #[inline]
  pub fn color_slot(&self) -> &CS::ColorTextures {
    &self.color_slot
  }

  #[inline]
  pub fn depth_slot(&self) -> &DS::DepthTexture {
    &self.depth_slot
  }
}

/// A framebuffer has a color slot. A color slot can either be empty (the *unit* type is used,`()`)
/// or several color formats.
pub unsafe trait ColorSlot<X, L, D>
where X: TextureDriver,
      L: Layerable,
      D: Dimensionable,
      D::Size: Copy {
  /// Textures associated with this color slot.
  type ColorTextures;

  /// Turn a color slot into a list of pixel formats.
  fn color_formats() -> Vec<PixelFormat>;

  /// Reify a list of raw textures.
  fn reify_textures<C, I>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    textures: &mut I,
  ) -> Self::ColorTextures
  where C: GraphicsContext<Driver = X>,
        I: Iterator<Item = X::Texture>;
}

unsafe impl<X, L, D> ColorSlot<X, L, D> for ()
where X: TextureDriver,
      L: Layerable,
      D: Dimensionable,
      D::Size: Copy {
  type ColorTextures = ();

  fn color_formats() -> Vec<PixelFormat> {
    Vec::new()
  }

  fn reify_textures<C, I>(
    _: &mut C,
    _: D::Size,
    _: usize,
    _: &mut I
  ) -> Self::ColorTextures
  where C: GraphicsContext<Driver = X>,
        I: Iterator<Item = X::Texture> {
    ()
  }
}

unsafe impl<X, L, D, P> ColorSlot<X, L, D> for P
where X: TextureDriver,
      L: Layerable,
      D: Dimensionable,
      D::Size: Copy,
      Self: ColorPixel + RenderablePixel {
  type ColorTextures = Texture<X, L, D, P>;

  fn color_formats() -> Vec<PixelFormat> {
    vec![P::pixel_format()]
  }

  fn reify_textures<C, I>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    textures: &mut I
  ) -> Self::ColorTextures
  where C: GraphicsContext<Driver = X>,
    I: Iterator<Item = X::Texture> {
    let color_texture = textures.next().unwrap();

    unsafe {
      let raw = RawTexture::new(
        ctx.state().clone(),
        color_texture,
        opengl_target(L::layering(), D::dim()),
      );
      Texture::from_raw(raw, size, mipmaps)
    }
  }
}

macro_rules! impl_color_slot_tuple {
  ($($pf:ident),*) => {
    unsafe impl<X, L, D, $($pf),*> ColorSlot<X, L, D> for ($($pf),*)
    where X: TextureDriver,
          L: Layerable,
          D: Dimensionable,
          D::Size: Copy,
          $(
            $pf: ColorPixel + RenderablePixel
          ),* {
      type ColorTextures = ($(Texture<X, L, D, $pf>),*);

      fn color_formats() -> Vec<PixelFormat> {
        vec![$($pf::pixel_format()),*]
      }

      fn reify_textures<C, I>(
        ctx: &mut C,
        size: D::Size,
        mipmaps: usize,
        textures: &mut I
      ) -> Self::ColorTextures
      where C: GraphicsContext<Driver = X>,
            I: Iterator<Item = X::Texture> {
        ($($pf::reify_textures(ctx, size, mipmaps, textures)),*)
      }
    }
  }
}

// Recursive macro that implements ColorSlot for tuples in decreasing order.
macro_rules! impl_color_slot_tuples {
  ($first:ident , $second:ident) => {
    // stop at pairs
    impl_color_slot_tuple!($first, $second);
  };

  ($first:ident , $($pf:ident),*) => {
    // implement the same list without the first type (reduced by one)
    impl_color_slot_tuples!($($pf),*);
    // implement the current list
    impl_color_slot_tuple!($first, $($pf),*);
  };
}

impl_color_slot_tuples!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11);

/// A framebuffer has a depth slot. A depth slot can either be empty (the *unit* type is used, `()`)
/// or a single depth format.
pub unsafe trait DepthSlot<X, L, D>
where X: TextureDriver,
      L: Layerable,
      D: Dimensionable,
      D::Size: Copy {
  /// Texture associated with this color slot.
  type DepthTexture;

  /// Turn a depth slot into a pixel format.
  fn depth_format() -> Option<PixelFormat>;

  /// Reify a raw textures into a depth slot.
  fn reify_texture<C, T>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    texture: T
  ) -> Self::DepthTexture
  where C: GraphicsContext<Driver = X>,
        T: Into<Option<X::Texture>>;
}

unsafe impl<X, L, D> DepthSlot<X, L, D> for ()
where X: TextureeDriver,
      L: Layerable,
      D: Dimensionable,
      D::Size: Copy {
  type DepthTexture = ();

  fn depth_format() -> Option<PixelFormat> {
    None
  }

  fn reify_texture<C, T>(
    _: &mut C,
    _: D::Size,
    _: usize,
    _: T
  ) -> Self::DepthTexture
  where C: GraphicsContext<Driver = X>,
        T: Into<Option<X::Texture>> {
    ()
  }
}

unsafe impl<X, L, D, P> DepthSlot<X, L, D> for P
where X: TextureDriver,
      L: Layerable,
      D: Dimensionable,
      D::Size: Copy,
      P: DepthPixel {
  type DepthTexture = Texture<X, L, D, P>;

  fn depth_format() -> Option<PixelFormat> {
    Some(P::pixel_format())
  }

  fn reify_texture<C, T>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    texture: T
  ) -> Self::DepthTexture
  where C: GraphicsContext<Driver = X>,
        T: Into<Option<X::Texture>> {
    unsafe {
      let raw = RawTexture::new(
        ctx.state().clone(),
        texture.into().unwrap(),
        opengl_target(L::layering(), D::dim()),
      );
      Texture::from_raw(raw, size, mipmaps)
    }
  }
}
