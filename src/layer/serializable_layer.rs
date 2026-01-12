use crate::Tensor;
use crate::layer::Layer;

/// A trait for layers that can be serialized and deserialized.
pub trait SerializableLayer: Layer {
    /// Returns the type of the layer as a string.
    fn layer_type(&self) -> &'static str;
    /// Serializes the layer's configuration into a byte vector.
    fn serialize_config(&self) -> Vec<u8>;
    /// Loads the layer's configuration from a byte slice.
    fn load_config(&self, data: &[u8]);

    /// Returns the layer's parameters as a vector of (name, tensor) pairs.
    fn parameters_serializable(&self) -> Vec<(String, Tensor)>;
    /// Loads the layer's parameters from a vector of (name, tensor) pairs.
    fn load_parameters(&self, params: Vec<(String, Tensor)>);
}
