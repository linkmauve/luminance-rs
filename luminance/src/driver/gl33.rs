mod buffer;
mod render_state;
mod state;

use crate::driver::gl33::state::GraphicsState;
use std::cell::RefCell;
use std::rc::Rc;

pub struct GL33 {
  state: Rc<RefCell<GraphicsState>>
}
