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

impl TessDriver for GL33 {
  type Tess = ();

  type TessBuilder = TessBuilder;

  type Err = String;

  unsafe fn new_tess_builder(&mut self) -> Result<Self::TessBuilder, Self::Err> {
    Ok(TessBuilder {
      vertex_buffers: Vec::new(),
      index_buffer: None,
      instance_buffers: Vec::new()
    })
  }
}
