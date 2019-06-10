//! This module provides texture features.
//!
//! # Introduction to textures
//!
//! Textures are used intensively in graphics programs as they tend to be the *de facto* memory area
//! to store data. You use them typically when you want to customize a render, hold a render’s
//! texels or even store arbritrary data.
//!
//! Currently, the following textures are supported:
//!
//! - 1D, 2D and 3D textures
//! - cubemaps
//! - array of textures (any of the types above)
//!
//! Those combinations are encoded by several types. First of all, `Texture<L, D, P>` is the
//! polymorphic type used to represent textures. The `L` type variable is the *layering type* of
//! the texture. It can either be `Flat` or `Layered`. The `D` type variable is the dimension of the
//! texture. It can either be `Dim1`, `Dim2`, `Dim3` or `Cubemap`. Finally, the `P` type variable
//! is the pixel format the texture follows. See the `pixel` module for further details about pixel
//! formats.
//!
//! Additionally, all textures have between 0 or several *mipmaps*. Mipmaps are additional layers of
//! texels used to perform trilinear filtering in most applications. Those are low-definition images
//! of the the base image used to smoothly interpolate texels when a projection kicks in. See
//! [this](https://en.wikipedia.org/wiki/Mipmap) for more insight.
//!
//! # Creating textures
//!
//! Textures are created by providing a size, the number of mipmaps that should be used and a
//! reference to a `Sampler` object. Up to now, textures and samplers form the same object – but
//! that might change in the future. Samplers are just a way to describe how texels will be fetched
//! from a shader.
//!
//! ## Associated types
//!
//! Because textures might have different shapes, the types of their sizes and offsets vary. You
//! have to look at the implementation of `Dimensionable::Size` and `Dimensionable::Offset` to know
//! which type you have to pass. For instance, for a 2D texture – e.g. `Texture<Flat, Dim2, _>`, you
//! have to pass a pair `(width, height)`.
//!
//! ## Samplers
//!
//! Samplers gather filters – i.e. how a shader should interpolate texels while fetching them,
//! wrap rules – i.e. how a shader should behave when leaving the normalized UV coordinates? and
//! a depth comparison, for depth textures only. See the documentation of `Sampler` for further
//! explanations.
//!
//! Samplers must be declared in the shader code according to the type of the texture used in the
//! Rust code. The size won’t matter, only the type. Here’s an exhaustive type of which sampler type
//! you must use according to the type of pixel format ([`PixelFormat`]) you use:
//!
//! > The `*` must be replaced by the dimension you use for your texture. If you use `Dim2` for
//! > instance, replace with `2`, as in `sampler*D -> sampler2D`.
//!
//! | `PixelFormat` | GLSL sampler type |
//! |---------------|-------------------|
//! | `R8I`         | `isampler*D`      |
//! | `R8UI`        | `usampler*D`      |
//! | `R16I`        | `isampler*D`      |
//! | `R16UI`       | `usampler*D`      |
//! | `R32I`        | `isampler*D`      |
//! | `R32UI`       | `usampler*D`      |
//! | `R32F`        | `sampler*D`       |
//! | `RG8I`        | `isampler*D`      |
//! | `RG8UI`       | `usampler*D`      |
//! | `RG16I`       | `isampler*D`      |
//! | `RG16UI`      | `usampler*D`      |
//! | `RG32I`       | `isampler*D`      |
//! | `RG32UI`      | `usampler*D`      |
//! | `RG32F`       | `sampler*D`       |
//! | `RGB8I`       | `isampler*D`      |
//! | `RGB8UI`      | `usampler*D`      |
//! | `RGB16I`      | `isampler*D`      |
//! | `RGB16UI`     | `usampler*D`      |
//! | `RGB32I`      | `isampler*D`      |
//! | `RGB32UI`     | `usampler*D`      |
//! | `RGB32F`      | `sampler*D`       |
//! | `RGBA8I`      | `isampler*D`      |
//! | `RGBA8UI`     | `usampler*D`      |
//! | `RGBA16I`     | `isampler*D`      |
//! | `RGBA16UI`    | `usampler*D`      |
//! | `RGBA32I`     | `isampler*D`      |
//! | `RGBA32UI`    | `usampler*D`      |
//! | `RGBA32F`     | `sampler*D`       |
//! | `Depth32F`    | `sampler1D`       |
//!
//! # Uploading data to textures
//!
//! One of the primary use of textures is to store images so that they can be used in your
//! application mapped on objects in your scene, for instance. In order to do so, you have to load
//! the image from the disk – see the awesome [image](https://crates.io/crates/image) – and then
//! upload the data to the texture. You have several functions to do so:
//!
//! - `Texture::upload`: this function takes a slice of texels and upload them to the whole texture memory
//! - `Texture::upload_part`: this function does the same thing as `Texture::upload`, but gives you the extra
//!   control on where in the texture you want to upload and with which size
//! - `Texture::upload_raw`: this function takes a slice of raw encoding data and upload them to the whole
//!   texture memory. This is especially handy when your texture has several channels but the data you have
//!   don’t take channels into account and are just *raw* data.
//! - `Texture::upload_part_raw`: same thing as above, but with offset and size control.
//!
//! Alternatively, you can clear the texture with `Texture::clear` and `Texture::clear_part`.
//!
//! # Retrieving texels
//!
//! The function `Texel::get_raw_texels` must be used to retreive texels out of a texture. This
//! function allocates memory, so be careful when using it.

use std::cell::RefCell;
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::os::raw::c_void;
use std::ptr;
use std::rc::Rc;

use crate::context::GraphicsContext;
use crate::driver::TextureDriver;
use crate::pixel::{Pixel, PixelFormat};

/// How to wrap texture coordinates while sampling textures?
#[derive(Clone, Copy, Debug)]
pub enum Wrap {
  /// If textures coordinates lay outside of *[0;1]*, they will be clamped to either *0* or *1* for
  /// every components.
  ClampToEdge,
  /// Textures coordinates are repeated if they lay outside of *[0;1]*. Picture this as:
  ///
  /// ```ignore
  /// // given the frac function returning the fractional part of a floating number:
  /// coord_ith = frac(coord_ith); // always between [0;1]
  /// ```
  Repeat,
  /// Same as `Repeat` but it will alternatively repeat between *[0;1]* and *[1;0]*.
  MirroredRepeat,
}

/// Minification filter.
#[derive(Clone, Copy, Debug)]
pub enum MinFilter {
  /// Nearest interpolation.
  Nearest,
  /// Linear interpolation between surrounding pixels.
  Linear,
  /// This filter will select the nearest mipmap between two samples and will perform a nearest
  /// interpolation afterwards.
  NearestMipmapNearest,
  /// This filter will select the nearest mipmap between two samples and will perform a linear
  /// interpolation afterwards.
  NearestMipmapLinear,
  /// This filter will linearly interpolate between two mipmaps, which selected texels would have
  /// been interpolated with a nearest filter.
  LinearMipmapNearest,
  /// This filter will linearly interpolate between two mipmaps, which selected texels would have
  /// been linarily interpolated as well.
  LinearMipmapLinear,
}

/// Magnification filter.
#[derive(Clone, Copy, Debug)]
pub enum MagFilter {
  /// Nearest interpolation.
  Nearest,
  /// Linear interpolation between surrounding pixels.
  Linear,
}

/// Depth comparison to perform while depth test. `a` is the incoming fragment’s depth and b is the
/// fragment’s depth that is already stored.
#[derive(Clone, Copy, Debug)]
pub enum DepthComparison {
  /// Depth test never succeeds.
  Never,
  /// Depth test always succeeds.
  Always,
  /// Depth test succeeds if `a == b`.
  Equal,
  /// Depth test succeeds if `a != b`.
  NotEqual,
  /// Depth test succeeds if `a < b`.
  Less,
  /// Depth test succeeds if `a <= b`.
  LessOrEqual,
  /// Depth test succeeds if `a > b`.
  Greater,
  /// Depth test succeeds if `a >= b`.
  GreaterOrEqual,
}

/// Reify a type into a `Dim`.
pub trait Dimensionable {
  type Size: Copy;
  type Offset: Copy;

  /// Dimension.
  fn dim() -> Dim;

  /// Width of the associated `Size`.
  fn width(size: Self::Size) -> u32;

  /// Height of the associated `Size`. If it doesn’t have one, set it to 1.
  #[inline(always)]
  fn height(_: Self::Size) -> u32 {
    1
  }

  /// Depth of the associated `Size`. If it doesn’t have one, set it to 1.
  #[inline(always)]
  fn depth(_: Self::Size) -> u32 {
    1
  }

  /// X offset.
  fn x_offset(offset: Self::Offset) -> u32;

  /// Y offset. If it doesn’t have one, set it to 0.
  #[inline(always)]
  fn y_offset(_: Self::Offset) -> u32 {
    1
  }

  /// Z offset. If it doesn’t have one, set it to 0.
  #[inline(always)]
  fn z_offset(_: Self::Offset) -> u32 {
    1
  }

  /// Zero offset.
  fn zero_offset() -> Self::Offset;
}

// Capacity of the dimension, which is the product of the width, height and depth.
#[inline(always)]
fn dim_capacity<D>(size: D::Size) -> u32 where D: Dimensionable {
  D::width(size) * D::height(size) * D::depth(size)
}

/// Dimension of a texture.
#[derive(Clone, Copy, Debug)]
pub enum Dim {
  Dim1,
  Dim2,
  Dim3,
  Cubemap,
}

/// 1D dimension.
#[derive(Clone, Copy, Debug)]
pub struct Dim1;

impl Dimensionable for Dim1 {
  type Offset = u32;
  type Size = u32;

  #[inline(always)]
  fn dim() -> Dim {
    Dim::Dim1
  }

  #[inline(always)]
  fn width(w: Self::Size) -> u32 {
    w
  }

  #[inline(always)]
  fn x_offset(off: Self::Offset) -> u32 {
    off
  }

  #[inline(always)]
  fn zero_offset() -> Self::Offset {
    0
  }
}

/// 2D dimension.
#[derive(Clone, Copy, Debug)]
pub struct Dim2;

impl Dimensionable for Dim2 {
  type Offset = [u32; 2];
  type Size = [u32; 2];

  #[inline(always)]
  fn dim() -> Dim {
    Dim::Dim2
  }

  #[inline(always)]
  fn width(size: Self::Size) -> u32 {
    size[0]
  }

  #[inline(always)]
  fn height(size: Self::Size) -> u32 {
    size[1]
  }

  #[inline(always)]
  fn x_offset(off: Self::Offset) -> u32 {
    off[0]
  }

  #[inline(always)]
  fn y_offset(off: Self::Offset) -> u32 {
    off[1]
  }

  #[inline(always)]
  fn zero_offset() -> Self::Offset {
    [0, 0]
  }
}

/// 3D dimension.
#[derive(Clone, Copy, Debug)]
pub struct Dim3;

impl Dimensionable for Dim3 {
  type Offset = [u32; 3];
  type Size = [u32; 3];

  #[inline(always)]
  fn dim() -> Dim {
    Dim::Dim3
  }

  #[inline(always)]
  fn width(size: Self::Size) -> u32 {
    size[0]
  }

  #[inline(always)]
  fn height(size: Self::Size) -> u32 {
    size[1]
  }

  #[inline(always)]
  fn depth(size: Self::Size) -> u32 {
    size[2]
  }

  #[inline(always)]
  fn x_offset(off: Self::Offset) -> u32 {
    off[0]
  }

  #[inline(always)]
  fn y_offset(off: Self::Offset) -> u32 {
    off[1]
  }

  #[inline(always)]
  fn z_offset(off: Self::Offset) -> u32 {
    off[2]
  }

  #[inline(always)]
  fn zero_offset() -> Self::Offset {
    [0, 0, 0]
  }
}

/// Cubemap dimension.
#[derive(Clone, Copy, Debug)]
pub struct Cubemap;

impl Dimensionable for Cubemap {
  type Offset = ([u32; 2], CubeFace);
  type Size = u32;

  #[inline(always)]
  fn dim() -> Dim {
    Dim::Cubemap
  }

  #[inline(always)]
  fn width(s: Self::Size) -> u32 {
    s
  }

  #[inline(always)]
  fn height(s: Self::Size) -> u32 {
    s
  }

  #[inline(always)]
  fn depth(_: Self::Size) -> u32 {
    6
  }

  #[inline(always)]
  fn x_offset(off: Self::Offset) -> u32 {
    off.0[0]
  }

  #[inline(always)]
  fn y_offset(off: Self::Offset) -> u32 {
    off.0[1]
  }

  #[inline(always)]
  fn z_offset(off: Self::Offset) -> u32 {
    match off.1 {
      CubeFace::PositiveX => 0,
      CubeFace::NegativeX => 1,
      CubeFace::PositiveY => 2,
      CubeFace::NegativeY => 3,
      CubeFace::PositiveZ => 4,
      CubeFace::NegativeZ => 5,
    }
  }

  #[inline(always)]
  fn zero_offset() -> Self::Offset {
    ([0, 0], CubeFace::PositiveX)
  }
}

/// Faces of a cubemap.
#[derive(Clone, Copy, Debug)]
pub enum CubeFace {
  PositiveX,
  NegativeX,
  PositiveY,
  NegativeY,
  PositiveZ,
  NegativeZ,
}

/// Trait used to reify a type into a `Layering`.
pub trait Layerable {
  /// Reify to `Layering`.
  fn layering() -> Layering;
}

/// Texture layering. If a texture is layered, it has an extra coordinate to access the layer.
#[derive(Clone, Copy, Debug)]
pub enum Layering {
  /// Non-layered.
  Flat,
  /// Layered.
  Layered,
}

/// Flat texture hint.
///
/// A flat texture means it doesn’t have the concept of layers.
#[derive(Clone, Copy, Debug)]
pub struct Flat;

impl Layerable for Flat {
  #[inline(always)]
  fn layering() -> Layering {
    Layering::Flat
  }
}

/// Layered texture hint.
///
/// A layered texture has an extra coordinate to access the layer and can be thought of as an array
/// of textures.
#[derive(Clone, Copy, Debug)]
pub struct Layered;

impl Layerable for Layered {
  #[inline(always)]
  fn layering() -> Layering {
    Layering::Layered
  }
}

/// Texture.
///
/// `L` refers to the layering type; `D` refers to the dimension; `P` is the pixel format for the
/// texels.
pub struct Texture<X, L, D, P>
where X: ?Sized + TextureDriver,
      L: Layerable,
      D: Dimensionable,
      P: Pixel, {
  raw: X::Texture,
  size: D::Size,
  mipmaps: usize, // number of mipmaps
  _l: PhantomData<L>,
  _p: PhantomData<P>,
}

impl<X, L, D, P> Deref for Texture<X, L, D, P>
where X: ?Sized + TextureDriver,
      L: Layerable,
      D: Dimensionable,
      P: Pixel, {
  type Target = X::Texture;

  #[inline(always)]
  fn deref(&self) -> &Self::Target {
    &self.raw
  }
}

impl<X, L, D, P> DerefMut for Texture<X, L, D, P>
where X: ?Sized + TextureDriver,
      L: Layerable,
      D: Dimensionable,
      P: Pixel, {
  #[inline(always)]
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.raw
  }
}

impl<X, L, D, P> Drop for Texture<X, L, D, P>
where X: ?Sized + TextureDriver,
      L: Layerable,
      D: Dimensionable,
      P: Pixel, {
  #[inline(always)]
  fn drop(&mut self) {
    unsafe { X::drop_texture(&mut self.raw); }
  }
}

impl<X, L, D, P> Texture<X, L, D, P>
where X: ?Sized + TextureDriver,
      L: Layerable,
      D: Dimensionable,
      P: Pixel, {
  pub fn new<C>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    sampler: &Sampler
  ) -> Result<Self, X::Err>
  where C: GraphicsContext<Driver = X> {
    let mipmaps = mipmaps + 1; // + 1 prevent having 0 mipmaps
    let raw = unsafe { ctx.driver().new_texture::<L, D, P>(size, mipmaps, sampler) };

    raw.map(|raw| Texture {
      raw,
      size,
      mipmaps,
      _l: PhantomData,
      _p: PhantomData,
    })
  }

  /// Create a texture from its backend representation.
  pub(crate) unsafe fn from_raw(raw: X::Texture, size: D::Size, mipmaps: usize) -> Self {
    Texture {
      raw,
      size,
      mipmaps: mipmaps + 1,
      _l: PhantomData,
      _p: PhantomData,
    }
  }

  /// Convert a texture to its raw representation.
  pub fn to_raw(mut self) -> X::Texture {
    let raw = mem::replace(&mut self.raw, unsafe { mem::uninitialized() });

    // forget self so that we don’t call drop on it after the function has returned
    mem::forget(self);
    raw
  }

  /// Number of mipmaps in the texture.
  #[inline(always)]
  pub fn mipmaps(&self) -> usize {
    self.mipmaps
  }

  /// Clear a part of a texture.
  ///
  /// The part being cleared is defined by a rectangle in which the `offset` represents the
  /// left-upper corner and the `size` gives the dimension of the rectangle. All the covered texels
  /// by this rectangle will be cleared to the `pixel` value.
  #[inline(always)]
  pub fn clear_part(
    &mut self,
    gen_mipmaps: bool,
    offset: D::Offset,
    size: D::Size,
    pixel: P::Encoding
  ) -> Result<(), X::Err>
  where P::Encoding: Copy {
    self.upload_part(
      gen_mipmaps,
      offset,
      size,
      &vec![pixel; dim_capacity::<D>(size) as usize],
    )
  }

  /// Clear a whole texture with a `pixel` value.
  #[inline(always)]
  pub fn clear(
    &mut self,
    gen_mipmaps: bool,
    pixel: P::Encoding
  ) -> Result<(), X::Err>
  where P::Encoding: Copy {
    self.clear_part(gen_mipmaps, D::zero_offset(), self.size, pixel)
  }

  /// Upload texels to a part of a texture.
  ///
  /// The part being updated is defined by a rectangle in which the `offset` represents the
  /// left-upper corner and the `size` gives the dimension of the rectangle. All the covered texels
  /// by this rectangle will be updated by the `texels` slice.
  pub fn upload_part(
    &mut self,
    gen_mipmaps: bool, // TODO: proper typing instead of bool
    offset: D::Offset,
    size: D::Size,
    texels: &[P::Encoding],
  ) -> Result<(), X::Err> {
    unsafe { X::upload_part::<L, D, P>(&mut self.raw, gen_mipmaps, offset, size, texels) }
  }

  /// Upload `texels` to the whole texture.
  #[inline(always)]
  pub fn upload(
    &mut self,
    gen_mipmaps: bool, // FIXME: bool typing
    texels: &[P::Encoding],
  ) -> Result<(), X::Err> {
    self.upload_part(gen_mipmaps, D::zero_offset(), self.size, texels)
  }

  /// Upload raw `texels` to a part of a texture.
  ///
  /// This function is similar to `upload_part` but it works on `P::RawEncoding` instead of
  /// `P::Encoding`. This useful when the texels are represented as a contiguous array of raw
  /// components of the texels.
  pub fn upload_part_raw(
    &mut self,
    gen_mipmaps: bool,
    offset: D::Offset,
    size: D::Size,
    texels: &[P::RawEncoding],
  ) -> Result<(), X::Err> {
    unsafe { X::upload_part_raw::<L, D, P>(&mut self.raw, gen_mipmaps, offset, size, texels) }
  }

  /// Upload raw `texels` to the whole texture.
  #[inline(always)]
  pub fn upload_raw(
    &mut self,
    gen_mipmaps: bool,
    texels: &[P::RawEncoding]
  ) -> Result<(), X::Err> {
    self.upload_part_raw(gen_mipmaps, D::zero_offset(), self.size, texels)
  }

  // FIXME: cubemaps?
  /// Get the raw texels associated with this texture.
  pub fn get_raw_texels(&self) -> Result<Vec<P::RawEncoding>, X::Err>
  where P: Pixel,
        P::RawEncoding: Copy, {
    unsafe { X::get_raw_texels::<P>(&self.raw) }
  }

  #[inline(always)]
  pub fn size(&self) -> D::Size {
    self.size
  }
}

/// A `Sampler` object gives hint on how a `Texture` should be sampled.
#[derive(Clone, Copy, Debug)]
pub struct Sampler {
  /// How should we wrap around the *r* sampling coordinate?
  pub wrap_r: Wrap,
  /// How should we wrap around the *s* sampling coordinate?
  pub wrap_s: Wrap,
  /// How should we wrap around the *t* sampling coordinate?
  pub wrap_t: Wrap,
  /// Minification filter.
  pub min_filter: MinFilter,
  /// Magnification filter.
  pub mag_filter: MagFilter,
  /// For depth textures, should we perform depth comparison and if so, how?
  pub depth_comparison: Option<DepthComparison>,
}

/// Default value is as following:
impl Default for Sampler {
  fn default() -> Self {
    Sampler {
      wrap_r: Wrap::ClampToEdge,
      wrap_s: Wrap::ClampToEdge,
      wrap_t: Wrap::ClampToEdge,
      min_filter: MinFilter::NearestMipmapLinear,
      mag_filter: MagFilter::Linear,
      depth_comparison: None,
    }
  }
}
