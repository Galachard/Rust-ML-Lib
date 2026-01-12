use crate::layer::SerializableLayer;
use crate::{Parameter, Tensor};

/// Trait representing a neural network layer.
/// Defines the essential methods that any layer must implement.
/// Includes methods for forward propagation and parameter retrieval.
/// Also provides an optional method for serialization support.
pub trait Layer {
    /// Performs the forward pass of the layer.
    /// Takes an input tensor and returns the output tensor after applying the layer's transformation.
    /// # Arguments
    /// * `input` - A reference to the input tensor.
    /// # Returns
    /// * A tensor representing the output after the layer's computation.
    fn forward(&self, input: &Tensor) -> Tensor;
    /// Retrieves the parameters of the layer.
    /// # Returns
    /// * A vector containing the layer's parameters.
    fn parameters(&self) -> Vec<Parameter>;
    /// Provides an optional reference to a serializable version of the layer.
    /// By default, this method returns `None`, indicating that the layer does not support serialization
    /// unless overridden by the implementing struct.
    /// # Returns
    /// * An optional reference to a `SerializableLayer`.
    fn as_serializable(&self) -> Option<&dyn SerializableLayer> {
        None
    }
}
