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

enum TessError {
  UnderlyingBufferError(BufferError<GL33>)
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

struct Tess {
  // OpenGL mode.
  mode: GLenum,
  // Vertex array object.
  vao: GLenum
}

impl TessDriver for GL33 {
  type Tess = ();

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

  unsafe fn drop_tess(tess: &mut Self::Tess) {
    gl::DeleteVertexArrays(1, &self.vao);
  }
}
