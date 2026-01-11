use crate::layer::SerializableLayer;
use crate::{Parameter, Tensor};

pub trait Layer {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Parameter>;
    fn as_serializable(&self) -> Option<&dyn SerializableLayer> {
        None
    }
}
