use crate::driver::TextureDriver;
use crate::driver::gl33::GL33;
use crate::driver::gl33::state::GraphicsState;
use crate::pixel::{Format, Pixel, PixelFormat, Size, Type};
use crate::texture::{
  DepthComparison, Dim, Dimensionable, Layerable, Layering, MagFilter, MinFilter, Sampler, Wrap
};
use gl;
use gl::types::*;
use std::cell::RefCell;
use std::fmt;
use std::mem::uninitialized;
use std::os::raw::c_void;
use std::ptr;
use std::rc::Rc;

pub struct RawTexture {
  handle: GLuint, // handle to the GPU texture object
  target: GLenum, // “type” of the texture; used for bindings
  state: Rc<RefCell<GraphicsState>>,
}

impl RawTexture {
  unsafe fn new(state: Rc<RefCell<GraphicsState>>, handle: GLuint, target: GLenum) -> Self {
    RawTexture {
      handle,
      target,
      state,
    }
  }

  #[inline]
  pub(crate) fn handle(&self) -> GLuint {
    self.handle
  }

  #[inline]
  pub(crate) fn target(&self) -> GLenum {
    self.target
  }
}

#[derive(Debug)]
pub enum TextureError {
  TextureStorageCreationFailed(String),
}

impl fmt::Display for TextureError {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      TextureError::TextureStorageCreationFailed(ref e) => {
        write!(f, "texture storage creation failed: {}", e)
      }
    }
  }
}

unsafe impl TextureDriver for GL33 {
  type Texture = RawTexture;

  type Err = TextureError;

  unsafe fn new_texture<L, D, P>(
    &mut self,
    size: D::Size,
    mipmaps: usize,
    sampler: &Sampler
  ) -> Result<Self::Texture, Self::Err>
  where L: Layerable,
        D: Dimensionable,
        P: Pixel {
    let mut texture = 0;
    let target = opengl_target(L::layering(), D::dim());

    gl::GenTextures(1, &mut texture);
    self.state.borrow_mut().bind_texture(target, texture);

    create_texture::<L, D>(target, size, mipmaps, P::pixel_format(), sampler)?;

    Ok(RawTexture::new(self.state.clone(), texture, target))
  }

  unsafe fn drop_texture(texture: &mut Self::Texture) {
    gl::DeleteTextures(1, &texture.handle)
  }

  unsafe fn upload_part<L, D, P>(
    texture: &mut Self::Texture,
    gen_mipmaps: bool,
    offset: D::Offset,
    size: D::Size,
    texels: &[P::Encoding]
  ) -> Result<(), Self::Err>
  where L: Layerable,
        D: Dimensionable,
        P: Pixel {
    let mut gfx_state = texture.state.borrow_mut();

    gfx_state.bind_texture(texture.target, texture.handle);

    upload_texels::<L, D, P, P::Encoding>(texture.target, offset, size, texels);

    if gen_mipmaps {
      gl::GenerateMipmap(texture.target);
    }

    gfx_state.bind_texture(texture.target, 0);

    Ok(())
  }

  unsafe fn upload_part_raw<L, D, P>(
    texture: &mut Self::Texture,
    gen_mipmaps: bool,
    offset: D::Offset,
    size: D::Size,
    texels: &[P::RawEncoding]
  ) -> Result<(), Self::Err>
  where L: Layerable,
        D: Dimensionable,
        P: Pixel {
    let mut gfx_state = texture.state.borrow_mut();

    gfx_state.bind_texture(texture.target, texture.handle);

    upload_texels::<L, D, P, P::RawEncoding>(texture.target, offset, size, texels);

    if gen_mipmaps {
      gl::GenerateMipmap(texture.target);
    }

    gfx_state.bind_texture(texture.target, 0);

    Ok(())
  }

  unsafe fn get_raw_texels<P>(
    texture: &Self::Texture
  ) -> Result<Vec<P::RawEncoding>, Self::Err>
  where P: Pixel,
        P::RawEncoding: Copy {
    let mut texels = Vec::new();
    let pf = P::pixel_format();
    let (format, _, ty) = opengl_pixel_format(pf).unwrap();

    let mut w = 0;
    let mut h = 0;

    let mut gfx_state = texture.state.borrow_mut();
    gfx_state.bind_texture(texture.target, texture.handle);

    // retrieve the size of the texture (w and h)
    gl::GetTexLevelParameteriv(texture.target, 0, gl::TEXTURE_WIDTH, &mut w);
    gl::GetTexLevelParameteriv(texture.target, 0, gl::TEXTURE_HEIGHT, &mut h);

    // resize the vec to allocate enough space to host the returned texels
    texels.resize((w * h) as usize * pixel_components(pf), uninitialized());

    gl::GetTexImage(texture.target, 0, format, ty, texels.as_mut_ptr() as *mut c_void);

    gfx_state.bind_texture(texture.target, 0);

    Ok(texels)
  }
}

pub(crate) fn opengl_target(l: Layering, d: Dim) -> GLenum {
  match l {
    Layering::Flat => match d {
      Dim::Dim1 => gl::TEXTURE_1D,
      Dim::Dim2 => gl::TEXTURE_2D,
      Dim::Dim3 => gl::TEXTURE_3D,
      Dim::Cubemap => gl::TEXTURE_CUBE_MAP,
    },
    Layering::Layered => match d {
      Dim::Dim1 => gl::TEXTURE_1D_ARRAY,
      Dim::Dim2 => gl::TEXTURE_2D_ARRAY,
      Dim::Dim3 => panic!("3D textures array not supported"),
      Dim::Cubemap => gl::TEXTURE_CUBE_MAP_ARRAY,
    },
  }
}

fn create_texture<L, D>(
  target: GLenum,
  size: D::Size,
  mipmaps: usize,
  pf: PixelFormat,
  sampler: &Sampler,
) -> Result<(), TextureError>
where
  L: Layerable,
  D: Dimensionable,
{
  set_texture_levels(target, mipmaps);
  apply_sampler_to_texture(target, sampler);
  create_texture_storage::<L, D>(size, mipmaps, pf)
}

fn set_texture_levels(target: GLenum, mipmaps: usize) {
  unsafe {
    gl::TexParameteri(target, gl::TEXTURE_BASE_LEVEL, 0);
    gl::TexParameteri(target, gl::TEXTURE_MAX_LEVEL, mipmaps as GLint - 1);
  }
}

fn apply_sampler_to_texture(target: GLenum, sampler: &Sampler) {
  unsafe {
    gl::TexParameteri(target, gl::TEXTURE_WRAP_R, opengl_wrap(sampler.wrap_r) as GLint);
    gl::TexParameteri(target, gl::TEXTURE_WRAP_S, opengl_wrap(sampler.wrap_s) as GLint);
    gl::TexParameteri(target, gl::TEXTURE_WRAP_T, opengl_wrap(sampler.wrap_t) as GLint);
    gl::TexParameteri(
      target,
      gl::TEXTURE_MIN_FILTER,
      opengl_min_filter(sampler.min_filter) as GLint,
    );
    gl::TexParameteri(
      target,
      gl::TEXTURE_MAG_FILTER,
      opengl_mag_filter(sampler.mag_filter) as GLint,
    );
    match sampler.depth_comparison {
      Some(fun) => {
        gl::TexParameteri(
          target,
          gl::TEXTURE_COMPARE_FUNC,
          opengl_depth_comparison(fun) as GLint,
        );
        gl::TexParameteri(
          target,
          gl::TEXTURE_COMPARE_MODE,
          gl::COMPARE_REF_TO_TEXTURE as GLint,
        );
      }
      None => {
        gl::TexParameteri(target, gl::TEXTURE_COMPARE_MODE, gl::NONE as GLint);
      }
    }
  }
}

fn opengl_wrap(wrap: Wrap) -> GLenum {
  match wrap {
    Wrap::ClampToEdge => gl::CLAMP_TO_EDGE,
    Wrap::Repeat => gl::REPEAT,
    Wrap::MirroredRepeat => gl::MIRRORED_REPEAT,
  }
}

fn opengl_min_filter(filter: MinFilter) -> GLenum {
  match filter {
    MinFilter::Nearest => gl::NEAREST,
    MinFilter::Linear => gl::LINEAR,
    MinFilter::NearestMipmapNearest => gl::NEAREST_MIPMAP_NEAREST,
    MinFilter::NearestMipmapLinear => gl::NEAREST_MIPMAP_LINEAR,
    MinFilter::LinearMipmapNearest => gl::LINEAR_MIPMAP_NEAREST,
    MinFilter::LinearMipmapLinear => gl::LINEAR_MIPMAP_LINEAR,
  }
}

fn opengl_mag_filter(filter: MagFilter) -> GLenum {
  match filter {
    MagFilter::Nearest => gl::NEAREST,
    MagFilter::Linear => gl::LINEAR,
  }
}

fn opengl_depth_comparison(fun: DepthComparison) -> GLenum {
  match fun {
    DepthComparison::Never => gl::NEVER,
    DepthComparison::Always => gl::ALWAYS,
    DepthComparison::Equal => gl::EQUAL,
    DepthComparison::NotEqual => gl::NOTEQUAL,
    DepthComparison::Less => gl::LESS,
    DepthComparison::LessOrEqual => gl::LEQUAL,
    DepthComparison::Greater => gl::GREATER,
    DepthComparison::GreaterOrEqual => gl::GEQUAL,
  }
}

fn create_texture_storage<L, D>(size: D::Size, mipmaps: usize, pf: PixelFormat) -> Result<(), TextureError>
where
  L: Layerable,
  D: Dimensionable,
{
  match opengl_pixel_format(pf) {
    Some(glf) => {
      let (format, iformat, encoding) = glf;

      match (L::layering(), D::dim()) {
        // 1D texture
        (Layering::Flat, Dim::Dim1) => {
          create_texture_1d_storage(format, iformat, encoding, D::width(size), mipmaps);
          Ok(())
        }

        // 2D texture
        (Layering::Flat, Dim::Dim2) => {
          create_texture_2d_storage(
            format,
            iformat,
            encoding,
            D::width(size),
            D::height(size),
            mipmaps,
          );
          Ok(())
        }

        // 3D texture
        (Layering::Flat, Dim::Dim3) => {
          create_texture_3d_storage(
            format,
            iformat,
            encoding,
            D::width(size),
            D::height(size),
            D::depth(size),
            mipmaps,
          );
          Ok(())
        }

        // cubemap
        (Layering::Flat, Dim::Cubemap) => {
          create_cubemap_storage(format, iformat, encoding, D::width(size), mipmaps);
          Ok(())
        }

        _ => {
          Err(TextureError::TextureStorageCreationFailed(format!(
              "unsupported texture OpenGL pixel format: {:?}",
              glf
          )))
        }
      }
    }

    None => {
      Err(TextureError::TextureStorageCreationFailed(format!(
          "unsupported texture pixel format: {:?}",
          pf
      )))
    }
  }
}

fn opengl_pixel_format(pf: PixelFormat) -> Option<(GLenum, GLenum, GLenum)> {
  match (pf.format, pf.encoding) {
    (Format::R(Size::Eight), Type::Integral) => Some((gl::RED_INTEGER, gl::R8I, gl::BYTE)),
    (Format::R(Size::Eight), Type::Unsigned) => Some((gl::RED_INTEGER, gl::R8UI, gl::UNSIGNED_BYTE)),
    (Format::R(Size::Sixteen), Type::Integral) => Some((gl::RED_INTEGER, gl::R16I, gl::SHORT)),
    (Format::R(Size::Sixteen), Type::Unsigned) => Some((gl::RED_INTEGER, gl::R16UI, gl::UNSIGNED_SHORT)),
    (Format::R(Size::ThirtyTwo), Type::Integral) => Some((gl::RED_INTEGER, gl::R32I, gl::INT)),
    (Format::R(Size::ThirtyTwo), Type::Unsigned) => Some((gl::RED_INTEGER, gl::R32UI, gl::UNSIGNED_INT)),
    (Format::R(Size::ThirtyTwo), Type::Floating) => Some((gl::RED, gl::R32F, gl::FLOAT)),

    (Format::RG(Size::Eight, Size::Eight), Type::Integral) => Some((gl::RG_INTEGER, gl::RG8I, gl::BYTE)),
    (Format::RG(Size::Eight, Size::Eight), Type::Unsigned) => {
      Some((gl::RG_INTEGER, gl::RG8UI, gl::UNSIGNED_BYTE))
    }
    (Format::RG(Size::Sixteen, Size::Sixteen), Type::Integral) => {
      Some((gl::RG_INTEGER, gl::RG16I, gl::SHORT))
    }
    (Format::RG(Size::Sixteen, Size::Sixteen), Type::Unsigned) => {
      Some((gl::RG_INTEGER, gl::RG16UI, gl::UNSIGNED_SHORT))
    }
    (Format::RG(Size::ThirtyTwo, Size::ThirtyTwo), Type::Integral) => {
      Some((gl::RG_INTEGER, gl::RG32I, gl::INT))
    }
    (Format::RG(Size::ThirtyTwo, Size::ThirtyTwo), Type::Unsigned) => {
      Some((gl::RG_INTEGER, gl::RG32UI, gl::UNSIGNED_INT))
    }
    (Format::RG(Size::ThirtyTwo, Size::ThirtyTwo), Type::Floating) => Some((gl::RG, gl::RG32F, gl::FLOAT)),

    (Format::RGB(Size::Eight, Size::Eight, Size::Eight), Type::Integral) => {
      Some((gl::RGB_INTEGER, gl::RGB8I, gl::BYTE))
    }
    (Format::RGB(Size::Eight, Size::Eight, Size::Eight), Type::Unsigned) => {
      Some((gl::RGB_INTEGER, gl::RGB8UI, gl::UNSIGNED_BYTE))
    }
    (Format::RGB(Size::Sixteen, Size::Sixteen, Size::Sixteen), Type::Integral) => {
      Some((gl::RGB_INTEGER, gl::RGB16I, gl::SHORT))
    }
    (Format::RGB(Size::Sixteen, Size::Sixteen, Size::Sixteen), Type::Unsigned) => {
      Some((gl::RGB_INTEGER, gl::RGB16UI, gl::UNSIGNED_SHORT))
    }
    (Format::RGB(Size::Eleven, Size::Eleven, Size::Ten), Type::Floating) => {
      Some((gl::RGB, gl::R11F_G11F_B10F, gl::FLOAT))
    }
    (Format::RGB(Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo), Type::Integral) => {
      Some((gl::RGB_INTEGER, gl::RGB32I, gl::INT))
    }
    (Format::RGB(Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo), Type::Unsigned) => {
      Some((gl::RGB_INTEGER, gl::RGB32UI, gl::UNSIGNED_INT))
    }
    (Format::RGB(Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo), Type::Floating) => {
      Some((gl::RGB, gl::RGB32F, gl::FLOAT))
    }

    (Format::RGBA(Size::Eight, Size::Eight, Size::Eight, Size::Eight), Type::Integral) => {
      Some((gl::RGBA_INTEGER, gl::RGBA8I, gl::BYTE))
    }
    (Format::RGBA(Size::Eight, Size::Eight, Size::Eight, Size::Eight), Type::Unsigned) => {
      Some((gl::RGBA_INTEGER, gl::RGBA8UI, gl::UNSIGNED_BYTE))
    }
    (Format::RGBA(Size::Sixteen, Size::Sixteen, Size::Sixteen, Size::Sixteen), Type::Integral) => {
      Some((gl::RGBA_INTEGER, gl::RGBA16I, gl::SHORT))
    }
    (Format::RGBA(Size::Sixteen, Size::Sixteen, Size::Sixteen, Size::Sixteen), Type::Unsigned) => {
      Some((gl::RGBA_INTEGER, gl::RGBA16UI, gl::UNSIGNED_SHORT))
    }
    (Format::RGBA(Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo), Type::Integral) => {
      Some((gl::RGBA_INTEGER, gl::RGBA32I, gl::INT))
    }
    (Format::RGBA(Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo), Type::Unsigned) => {
      Some((gl::RGBA_INTEGER, gl::RGBA32UI, gl::UNSIGNED_INT))
    }
    (Format::RGBA(Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo), Type::Floating) => {
      Some((gl::RGBA, gl::RGBA32F, gl::FLOAT))
    }

    (Format::Depth(Size::ThirtyTwo), Type::Floating) => {
      Some((gl::DEPTH_COMPONENT, gl::DEPTH_COMPONENT32F, gl::FLOAT))
    }

    _ => panic!("unsupported pixel format {:?}", pf),
  }
}

fn create_texture_1d_storage(format: GLenum, iformat: GLenum, encoding: GLenum, w: u32, mipmaps: usize) {
  for level in 0..mipmaps {
    let w = w / 2u32.pow(level as u32);

    unsafe {
      gl::TexImage1D(
        gl::TEXTURE_1D,
        level as GLint,
        iformat as GLint,
        w as GLsizei,
        0,
        format,
        encoding,
        ptr::null(),
      )
    };
  }
}

fn create_texture_2d_storage(
  format: GLenum,
  iformat: GLenum,
  encoding: GLenum,
  w: u32,
  h: u32,
  mipmaps: usize,
) {
  for level in 0..mipmaps {
    let div = 2u32.pow(level as u32);
    let w = w / div;
    let h = h / div;

    unsafe {
      gl::TexImage2D(
        gl::TEXTURE_2D,
        level as GLint,
        iformat as GLint,
        w as GLsizei,
        h as GLsizei,
        0,
        format,
        encoding,
        ptr::null(),
      )
    };
  }
}

fn create_texture_3d_storage(
  format: GLenum,
  iformat: GLenum,
  encoding: GLenum,
  w: u32,
  h: u32,
  d: u32,
  mipmaps: usize,
) {
  for level in 0..mipmaps {
    let div = 2u32.pow(level as u32);
    let w = w / div;
    let h = h / div;
    let d = d / div;

    unsafe {
      gl::TexImage3D(
        gl::TEXTURE_3D,
        level as GLint,
        iformat as GLint,
        w as GLsizei,
        h as GLsizei,
        d as GLsizei,
        0,
        format,
        encoding,
        ptr::null(),
      )
    };
  }
}

fn create_cubemap_storage(format: GLenum, iformat: GLenum, encoding: GLenum, s: u32, mipmaps: usize) {
  for level in 0..mipmaps {
    let s = s / 2u32.pow(level as u32);

    unsafe {
      gl::TexImage2D(
        gl::TEXTURE_CUBE_MAP,
        level as GLint,
        iformat as GLint,
        s as GLsizei,
        s as GLsizei,
        0,
        format,
        encoding,
        ptr::null(),
      )
    };
  }
}

fn upload_texels<L, D, P, T>(target: GLenum, off: D::Offset, size: D::Size, texels: &[T])
where
  L: Layerable,
  D: Dimensionable,
  P: Pixel,
{
  let pf = P::pixel_format();

  match opengl_pixel_format(pf) {
    Some((format, _, encoding)) => match L::layering() {
      Layering::Flat => match D::dim() {
        Dim::Dim1 => unsafe {
          gl::TexSubImage1D(
            target,
            0,
            D::x_offset(off) as GLint,
            D::width(size) as GLsizei,
            format,
            encoding,
            texels.as_ptr() as *const c_void,
          )
        },
        Dim::Dim2 => unsafe {
          gl::TexSubImage2D(
            target,
            0,
            D::x_offset(off) as GLint,
            D::y_offset(off) as GLint,
            D::width(size) as GLsizei,
            D::height(size) as GLsizei,
            format,
            encoding,
            texels.as_ptr() as *const c_void,
          )
        },
        Dim::Dim3 => unsafe {
          gl::TexSubImage3D(
            target,
            0,
            D::x_offset(off) as GLint,
            D::y_offset(off) as GLint,
            D::z_offset(off) as GLint,
            D::width(size) as GLsizei,
            D::height(size) as GLsizei,
            D::depth(size) as GLsizei,
            format,
            encoding,
            texels.as_ptr() as *const c_void,
          )
        },
        Dim::Cubemap => unsafe {
          gl::TexSubImage3D(
            target,
            0,
            D::x_offset(off) as GLint,
            D::y_offset(off) as GLint,
            (gl::TEXTURE_CUBE_MAP_POSITIVE_X + D::z_offset(off)) as GLint,
            D::width(size) as GLsizei,
            D::width(size) as GLsizei,
            1,
            format,
            encoding,
            texels.as_ptr() as *const c_void,
          )
        },
      },
      Layering::Layered => panic!("Layering::Layered not implemented yet"),
    },
    None => panic!("unknown pixel format"),
  }
}

// Return the number of components.
fn pixel_components(pf: PixelFormat) -> usize {
  match pf.format {
    Format::RGB(_, _, _) => 3,
    Format::RGBA(_, _, _, _) => 4,
    Format::Depth(_) => 1,
    _ => panic!("unsupported pixel format"),
  }
}
