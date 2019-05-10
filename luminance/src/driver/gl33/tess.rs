use gl;
use gl::types::*;

use crate::buffer::BufferError;
use crate::driver::TessDriver;
use crate::driver::gl33::GL33;
use crate::driver::gl33::buffer::RawBuffer;
use crate::tess::TessIndexType;
use crate::vertex::VertexDesc;

struct TessBuilder {
  vertex_buffers: Vec<VertexBuffer>,
  index_buffer: Option<IndexBuffer>,
  instance_buffers: Vec<VertexBuffer>,
}

impl TessBuilder {
  /// Guess how many vertices there are to render based on the current tessellation configuration or
  /// fail if incorrectly configured.
  fn guess_vert_nb_or_fail(&self, vert_nb: usize) -> Result<usize, TessError> {
    if vert_nb == 0 {
      // we don’t have an explicit vertex number to render; go and guess!
      if let Some(ref index_buffer) = self.index_buffer {
        // we have an index buffer: just use its size
        Ok(index_buffer.0.len())
      } else {
        // deduce the number of vertices based on the vertex buffers; they all
        // must be of the same length, otherwise it’s an error
        match self.vertex_buffers.len() {
          0 => {
            Err(TessError::AttributelessError("attributeless render with no vertex number".to_owned()))
          }

          1 => {
            Ok(self.vertex_buffers[0].buf.len())
          }

          _ => {
            let vert_nb = self.vertex_buffers[0].buf.len();
            let incoherent = Self::check_incoherent_buffers(self.vertex_buffers.iter(), vert_nb);

            if incoherent {
              Err(TessError::LengthIncoherency(vert_nb))
            } else {
              Ok(vert_nb)
            }
          }
        }
      }
    } else {
      // we have an explicit number of vertices to render, but we’re gonna check that number actually
      // makes sense
      if let Some(ref index_buffer) = self.index_buffer {
        // we have indices (indirect draw); so we’ll compare to them
        if index_buffer.0.len() < vert_nb {
          return Err(TessError::Overflow(index_buffer.0.len(), vert_nb));
        }
      } else {
        let incoherent = Self::check_incoherent_buffers(self.vertex_buffers.iter(), vert_nb);

        if incoherent {
          return Err(TessError::LengthIncoherency(vert_nb));
        } else if !self.vertex_buffers.is_empty() && self.vertex_buffers[0].buf.len() < vert_nb {
          return Err(TessError::Overflow(self.vertex_buffers[0].buf.len(), vert_nb));
        }
      }

      Ok(vert_nb)
    }
  }

  /// Guess how many instances there are to render based on the current configuration or fail if
  /// incorrectly configured.
  fn guess_inst_nb_or_fail(&self, inst_nb: usize) -> Result<usize, TessError> {
    if inst_nb == 0 {
      // we don’t have an explicit instance number to render; go and guess!
      // deduce the number of instances based on the instance buffers; they all must be of the same
      // length, otherwise it’s an error
      match self.instance_buffers.len() {
        0 => {
          // no instance buffer; we we’re not using instance rendering
          Ok(0)
        }

        1 => {
          Ok(self.instance_buffers[0].buf.len())
        }

        _ => {
          let inst_nb = self.instance_buffers[0].buf.len();
          let incoherent = Self::check_incoherent_buffers(self.instance_buffers.iter(), inst_nb);

          if incoherent {
            Err(TessError::LengthIncoherency(inst_nb))
          } else {
            Ok(inst_nb)
          }
        }
      }
    } else {
      // we have an explicit number of instances to render, but we’re gonna check that number
      // actually makes sense
      let incoherent = Self::check_incoherent_buffers(self.instance_buffers.iter(), self.vert_nb);

      if incoherent {
        return Err(TessError::LengthIncoherency(inst_nb));
      } else if !self.instance_buffers.is_empty() && self.instance_buffers[0].buf.len() < inst_nb {
        return Err(TessError::Overflow(self.instance_buffers[0].buf.len(), inst_nb));
      }

      Ok(inst_nb)
    }
  }

  /// Check whether any vertex buffer is incoherent in its length according to the input length.
  fn check_incoherent_buffers<'b, B>(
    mut buffers: B,
    len: usize
  ) -> bool
  where B: Iterator<Item = &'b VertexBuffer> {
    !buffers.all(|vb| vb.buf.len() == len)
  }

  /// Build a tessellation based on a given number of vertices to render by default.
  unsafe fn build_tess(self, vert_nb: usize, inst_nb: usize) -> Tess {
    let mut vao: GLuint = 0;
    let mut gfx_st = self.ctx.state().borrow_mut();

    gl::GenVertexArrays(1, &mut vao);

    gfx_st.bind_vertex_array(vao);

    // add the vertex buffers into the vao
    for vb in &self.vertex_buffers {
      gfx_st.bind_array_buffer(vb.buf.handle());
      set_vertex_pointers(&vb.fmt)
    }

    // in case of indexed render, bind the index buffer
    if let Some(ref index_buffer) = self.index_buffer {
      gfx_st.bind_element_array_buffer(index_buffer.0.handle());
    }

    // add instance buffers, if any
    for vb in &self.instance_buffers {
      gfx_st.bind_array_buffer(vb.buf.handle());
      set_vertex_pointers(&vb.fmt);
    }

    // create the index state, if any required
    let restart_index = self.restart_index;
    let index_state = self.index_buffer.map(move |(buffer, index_type)| {
      IndexedDrawState {
        _buf: buffer,
        restart_index,
        index_type: index_type.to_glenum(),
      }
    });

    // convert to OpenGL-friendly internals and return
    Tess {
      mode: self.mode.to_glenum(),
      vao,
      vert_nb,
      inst_nb,
      vertex_buffers: self.vertex_buffers,
      instance_buffers: self.instance_buffers,
      index_state,
    }
  }
}

enum TessError {
  UnderlyingBufferError(BufferError<GL33>),
  AttributelessError(String),
  LengthIncoherency(usize),
  Overflow(usize, usize),
}

struct VertexBuffer {
  // Indexed format of the buffer.
  fmt: VertexDesc,
  // Internal buffer.
  buf: RawBuffer
}

struct IndexBuffer {
  // Type of index.
  ty: TessIndexType,
  // Internal buffer.
  buf: RawBuffer
}

struct IndexedDrawState {
  _buf: RawBuffer
  restart_index: Option<u32>,
  index_type: GLenum
}

struct Tess {
  // OpenGL mode.
  mode: GLenum,
  // Vertex array object.
  vao: GLenum,
  // Number of vertices to render by default.
  vert_nb: usize,
  // Number of instances to render by default.
  inst_nb: usize,
  // Vertex buffers.
  vertex_buffers: Vec<VertexBuffer>,
  // Instances buffers.
  instance_buffers: Vec<VertexBuffer>,
  // Index state.
  index_state: Option<IndexedDrawState>,
}

impl TessDriver for GL33 {
  type Tess = Tess;

  type TessBuilder = TessBuilder;

  type Err = TessError;

  unsafe fn new_tess_builder(&mut self) -> Result<Self::TessBuilder, <Self as TessDriver>::Err> {
    Ok(TessBuilder {
      vertex_buffers: Vec::new(),
      index_buffer: None,
      instance_buffers: Vec::new()
    })
  }

  unsafe fn add_vertices<V>(
    &mut self,
    builder: &mut Self::TessBuilder,
    vertices: &[V]
  ) -> Result<(), Self::Err>
  where V: vertex::Vertex {
    let vb = VertexBuffer {
      fmt: V::vertex_desc(),
      buf: Buffer::from_slice_driver(self, vertices)
        .map_err(TessError::UnderlyingBufferError)?
        .to_driver_buf(),
    };

    builder.vertex_buffers.push(vb);

    Ok(())
  }

  unsafe fn add_instances<V>(
    &mut self,
    builder: &mut Self::TessBuilder,
    instances: &[V]
  ) -> Result<(), Self::Err>
  where V: vertex::Vertex {
    let vb = VertexBuffer {
      fmt: V::vertex_desc(),
      buf: Buffer::from_slice_driver(self, vertices)
        .map_err(TessError::UnderlyingBufferError)?
        .to_driver_buf(),
    };

    builder.instance_buffers.push(vb);

    Ok(())
  }

  unsafe fn set_indices<I>(
    &mut self,
    builder: &mut Self::TessBuilder,
    indices: &[I]
  ) -> Result<(), <Self as TessDriver>::Err>
  where I: tess::TessIndex {
    // create a new raw buffer containing the indices and turn it into a vertex buffer
    let buf = Buffer::from_slice_driver(self, indices)
      .map_err(TessError::UnderlyingBufferError)?
      .to_driver_buf();

    builder.index_buffer = Some((buf, I::INDEX_TYPE));

    Ok(())
  }

  unsafe fn build_tess(
    &mut self,
    builder: Self::TessBuilder,
    vert_nb: usize,
    inst_nb: usize
  ) -> Result<Self::Tess, <Self as TessDriver>::Err> {
    // try to deduce the number of vertices and instances to render if it’s not specified
    let vert_nb = builder.guess_vert_nb_or_fail(vert_nb)?;
    let inst_nb = builder.guess_inst_nb_or_fail(inst_nb)?;
    let tess = builder.build_tess(vert_nb, inst_nb);

    Ok(tess)
  }

  unsafe fn drop_tess(tess: &mut Self::Tess) {
    gl::DeleteVertexArrays(1, &self.vao);
  }

  unsafe fn tess_vertex_buffer<'a, V>(
    tess: &'a Self::Tess
  ) -> Result<&'a Self::Buffer, <Self as TessDriver>::Err>
  where V: vertex::Vertex {
    match tess.vertex_buffers.len() {
      0 => Err(TessMapError::ForbiddenAttributelessMapping),

      1 => {
        let vb = &tess.vertex_buffers[0];
        let target_fmt = V::vertex_desc(); // costs a bit

        if vb.fmt != target_fmt {
          Err(TessMapError::TypeMismatch(vb.fmt.clone(), target_fmt))
        } else {
          Ok(&vb.buf)
        }
      }

      _ => Err(TessMapError::ForbiddenDeinterleavedMapping),
    }
  }

  unsafe fn tess_vertex_buffer_mut<'a, V>(
    tess: &'a mut Self::Tess
  ) -> Result<&'a mut Self::Buffer, <Self as TessDriver>::Err>
  where V: vertex::Vertex {
    match tess.vertex_buffers.len() {
      0 => Err(TessMapError::ForbiddenAttributelessMapping),

      1 => {
        let vb = &mut tess.vertex_buffers[0];
        let target_fmt = V::vertex_desc(); // costs a bit

        if vb.fmt != target_fmt {
          Err(TessMapError::TypeMismatch(vb.fmt.clone(), target_fmt))
        } else {
          Ok(&mut vb.buf)
        }
      }

      _ => Err(TessMapError::ForbiddenDeinterleavedMapping),
    }
  }

  unsafe fn tess_inst_buffer<'a, V>(
    tess: &'a Self::Tess
  ) -> Result<&'a Self::Buffer, <Self as TessDriver>::Err>
  where V: vertex::Vertex {
    match tess.instance_buffers.len() {
      0 => Err(TessMapError::ForbiddenAttributelessMapping),

      1 => {
        let vb = &tess.instance_buffers[0];
        let target_fmt = V::vertex_desc(); // costs a bit

        if vb.fmt != target_fmt {
          Err(TessMapError::TypeMismatch(vb.fmt.clone(), target_fmt))
        } else {
          Ok(&vb.buf)
        }
      }

      _ => Err(TessMapError::ForbiddenDeinterleavedMapping),
    }
  }

  unsafe fn tess_inst_buffer_mut<'a, V>(
    tess: &'a mut Self::Tess
  ) -> Result<&'a mut Self::Buffer, <Self as TessDriver>::Err>
  where V: vertex::Vertex {
    match tess.instance_buffers.len() {
      0 => Err(TessMapError::ForbiddenAttributelessMapping),

      1 => {
        let vb = &mut tess.instance_buffers[0];
        let target_fmt = V::vertex_desc(); // costs a bit

        if vb.fmt != target_fmt {
          Err(TessMapError::TypeMismatch(vb.fmt.clone(), target_fmt))
        } else {
          Ok(&mut vb.buf)
        }
      }

      _ => Err(TessMapError::ForbiddenDeinterleavedMapping),
    }
  }

  unsafe fn render_tess(
    &mut self,
    tess: &Self::Tess,
    start_index: usize,
    vert_nb: usize,
    inst_nb: usize
  ) {
    let vert_nb = vert_nb as GLsizei;
    let inst_nb = inst_nb as GLsizei;

    unsafe {
      let mut gfx_st = self.state.borrow_mut();
      gfx_st.bind_vertex_array(tess.vao);

      if let Some(index_state) = tess.index_state.as_ref() {
        // indexed render
        let first = (index_state.index_type.bytes() * start_index) as *const c_void;

        if let Some(restart_index) = index_state.restart_index {
          gfx_st.set_vertex_restart(VertexRestart::On);
          gl::PrimitiveRestartIndex(restart_index);
        } else {
          gfx_st.set_vertex_restart(VertexRestart::Off);
        }

        if inst_nb <= 1 {
          gl::DrawElements(tess.mode, vert_nb, index_state.index_type.to_glenum(), first);
        } else {
          gl::DrawElementsInstanced(
            tess.mode,
            vert_nb,
            index_state.index_type.to_glenum(),
            first,
            inst_nb,
          );
        }
      } else {
        // direct render
        let first = start_index as GLint;

        if inst_nb <= 1 {
          gl::DrawArrays(tess.mode, first, vert_nb);
        } else {
          gl::DrawArraysInstanced(tess.mode, first, vert_nb, inst_nb);
        }
      }
    }
  }
}

// Give OpenGL types information on the content of the VBO by setting vertex descriptors and pointers
// to buffer memory.
fn set_vertex_pointers(descriptors: &VertexDesc) {
  // this function sets the vertex attribute pointer for the input list by computing:
  //   - The vertex attribute ID: this is the “rank” of the attribute in the input list (order
  //     matters, for short).
  //   - The stride: this is easily computed, since it’s the size (bytes) of a single vertex.
  //   - The offsets: each attribute has a given offset in the buffer. This is computed by
  //     accumulating the size of all previously set attributes.
  let offsets = aligned_offsets(descriptors);
  let vertex_weight = offset_based_vertex_weight(descriptors, &offsets) as GLsizei;

  for (desc, off) in descriptors.iter().zip(offsets) {
    set_component_format(vertex_weight, off, desc);
  }
}

// Compute offsets for all the vertex components according to the alignments provided.
fn aligned_offsets(descriptor: &VertexDesc) -> Vec<usize> {
  let mut offsets = Vec::with_capacity(descriptor.len());
  let mut off = 0;

  // compute offsets
  for desc in descriptor {
    let desc = &desc.attrib_desc;
    off = off_align(off, desc.align); // keep the current component descriptor aligned
    offsets.push(off);
    off += component_weight(desc); // increment the offset by the pratical size of the component
  }

  offsets
}

// Align an offset.
#[inline]
fn off_align(off: usize, align: usize) -> usize {
  let a = align - 1;
  (off + a) & !a
}

// Weight in bytes of a vertex component.
fn component_weight(f: &VertexAttribDesc) -> usize {
  dim_as_size(&f.dim) as usize * f.unit_size
}

fn dim_as_size(d: &VertexAttribDim) -> GLint {
  match *d {
    VertexAttribDim::Dim1 => 1,
    VertexAttribDim::Dim2 => 2,
    VertexAttribDim::Dim3 => 3,
    VertexAttribDim::Dim4 => 4,
  }
}

// Weight in bytes of a single vertex, taking into account padding so that the vertex stay correctly
// aligned.
fn offset_based_vertex_weight(descriptors: &VertexDesc, offsets: &[usize]) -> usize {
  if descriptors.is_empty() || offsets.is_empty() {
    return 0;
  }

  off_align(
    offsets[offsets.len() - 1] + component_weight(&descriptors[descriptors.len() - 1].attrib_desc),
    descriptors[0].attrib_desc.align,
  )
}

// Set the vertex component OpenGL pointers regarding the index of the component (i), the stride
fn set_component_format(stride: GLsizei, off: usize, desc: &VertexBufferDesc) {
  let attrib_desc = &desc.attrib_desc;
  let index = desc.index as GLuint;

  unsafe {
    match attrib_desc.ty {
      VertexAttribType::Floating => {
        gl::VertexAttribPointer(
          index,
          dim_as_size(&attrib_desc.dim),
          opengl_sized_type(&attrib_desc),
          gl::FALSE,
          stride,
          ptr::null::<c_void>().offset(off as isize),
          );
      },
      VertexAttribType::Integral | VertexAttribType::Unsigned | VertexAttribType::Boolean => {
        gl::VertexAttribIPointer(
          index,
          dim_as_size(&attrib_desc.dim),
          opengl_sized_type(&attrib_desc),
          stride,
          ptr::null::<c_void>().offset(off as isize),
          );
      },
    }

    // set vertex attribute divisor based on the vertex instancing configuration
    let divisor = match desc.instancing {
      VertexInstancing::On => 1,
      VertexInstancing::Off => 0
    };
    gl::VertexAttribDivisor(index, divisor);

    gl::EnableVertexAttribArray(index);
  }
}

fn opengl_sized_type(f: &VertexAttribDesc) -> GLenum {
  match (f.ty, f.unit_size) {
    (VertexAttribType::Integral, 1) => gl::BYTE,
    (VertexAttribType::Integral, 2) => gl::SHORT,
    (VertexAttribType::Integral, 4) => gl::INT,
    (VertexAttribType::Unsigned, 1) | (VertexAttribType::Boolean, 1) => gl::UNSIGNED_BYTE,
    (VertexAttribType::Unsigned, 2) => gl::UNSIGNED_SHORT,
    (VertexAttribType::Unsigned, 4) => gl::UNSIGNED_INT,
    (VertexAttribType::Floating, 4) => gl::FLOAT,
    _ => panic!("unsupported vertex component format: {:?}", f),
  }
}

impl Mode {
  fn to_glenum(self) -> GLenum {
    match self {
      Mode::Point => gl::POINTS,
      Mode::Line => gl::LINES,
      Mode::LineStrip => gl::LINE_STRIP,
      Mode::Triangle => gl::TRIANGLES,
      Mode::TriangleFan => gl::TRIANGLE_FAN,
      Mode::TriangleStrip => gl::TRIANGLE_STRIP,
    }
  }
}

impl TessIndexType {
  fn to_glenum(self) -> GLenum {
    match self {
      TessIndexType::U8 => gl::UNSIGNED_BYTE,
      TessIndexType::U16 => gl::UNSIGNED_SHORT,
      TessIndexType::U32 => gl::UNSIGNED_INT,
    }
  }

  fn bytes(self) -> usize {
    match self {
      TessIndexType::U8 => 1,
      TessIndexType::U16 => 2,
      TessIndexType::U32 => 4,
    }
  }
}
