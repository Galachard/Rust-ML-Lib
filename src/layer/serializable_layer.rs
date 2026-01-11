use crate::Tensor;
use crate::layer::Layer;

pub trait SerializableLayer: Layer {
    fn layer_type(&self) -> &'static str;
    fn serialize_config(&self) -> Vec<u8>;
    fn load_config(&self, data: &[u8]);

    fn parameters_serializable(&self) -> Vec<(String, Tensor)>;
    fn load_parameters(&self, params: Vec<(String, Tensor)>);
}
