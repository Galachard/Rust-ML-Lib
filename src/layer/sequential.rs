use crate::graph::Node;
use crate::layer::Layer;
use crate::optimizer::Optimizer;
use crate::{Parameter, Tensor};
use indicatif::{ProgressBar, ProgressStyle};

/// A sequential container for layers
/// Applies layers in the order they were added
/// Supports training loop with batching and progress bar
/// Supports serialization and deserialization of layers that implement SerializableLayer and
/// can be saved to and loaded from disk
/// Example:
/// ```
/// use ml_lib::layer::{Sequential, Linear, ELU, initializer::HeNormal, initializer::Zeros};
/// let mut model = Sequential::new();
/// model.add(Linear::new(784, 128, &HeNormal {}, &Zeros {}));
/// model.add(ELU::default());
/// model.add(Linear::new(128, 10, &HeNormal {}, &Zeros {}));
/// ```
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Layer for Sequential {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.layers
            .iter()
            .fold(input.clone(), |x, layer| layer.forward(&x))
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    // Not implementing as_serializable() for now.
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Sequential {
    pub fn new() -> Self {
        Self { layers: vec![] }
    }

    pub fn add<L: Layer + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }

    pub fn backward(&self, loss: &Tensor) {
        loss.backward();
    }

    pub fn step<O: Optimizer>(&self, opt: &mut O) {
        opt.step();
    }

    pub fn zero_grad<O: Optimizer>(&self, opt: &mut O) {
        opt.zero_grad();
    }

    /// Run training loop
    pub fn train(
        model: &Sequential,
        optimizer: &mut impl Optimizer,
        inputs: &[Tensor],
        targets: &[Tensor],
        epochs: usize,
        batch_size: usize,
        loss_function: impl Fn(&Tensor, &Tensor) -> Tensor,
    ) {
        assert_eq!(inputs.len(), targets.len());

        let n = inputs.len();

        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            let pb = ProgressBar::new((n / batch_size) as u64);
            pb.set_style(
                ProgressStyle::with_template(
                    "[{elapsed_precise}] {bar:32.white/grey} {pos}/{len} {msg}",
                )
                .unwrap(),
            );

            let mut i = 0;
            while i < n {
                let end = (i + batch_size).min(n);

                // Zero grad once per batch
                model.zero_grad(optimizer);

                for j in i..end {
                    let pred = model.forward(&inputs[j]);
                    let loss = loss_function(&pred, &targets[j]);

                    model.backward(&loss);
                    total_loss += loss.data[0];
                }
                // Apply accumulated gradients
                model.step(optimizer);
                i = end;

                pb.set_message(format!(
                    "Epoch {}/{}: loss={:.4}",
                    epoch + 1,
                    epochs,
                    total_loss / (end as f32)
                ));
                pb.inc(1);
            }

            pb.finish_with_message(format!(
                "Epoch {}/{}: loss={}",
                epoch + 1,
                epochs,
                total_loss / n as f32
            ));
        }
    }

    /// Serialize layers that implement SerializableLayer
    pub fn serialize(&self) -> Vec<u8> {
        let mut serialized = vec![];

        for layer in &self.layers {
            if let Some(sl) = layer.as_serializable() {
                // Layer type
                let t_bytes = sl.layer_type().as_bytes();
                serialized.push(t_bytes.len() as u8); // simple length prefix
                serialized.extend_from_slice(t_bytes);

                // Config
                let cfg_bytes = sl.serialize_config();
                let cfg_len = cfg_bytes.len() as u32;
                serialized.extend_from_slice(&cfg_len.to_le_bytes());
                serialized.extend(cfg_bytes);

                // Parameters
                let params = sl.parameters_serializable();
                serialized.push(params.len() as u8);
                for (name, tensor) in params {
                    let name_bytes = name.as_bytes();
                    serialized.push(name_bytes.len() as u8);
                    serialized.extend_from_slice(name_bytes);

                    // Tensor shape
                    let shape_len = tensor.shape.len() as u8;
                    serialized.push(shape_len);
                    for dim in &tensor.shape {
                        serialized.extend_from_slice(&(*dim as u32).to_le_bytes());
                    }

                    // Tensor data
                    let data_len = tensor.data.len() as u32;
                    serialized.extend_from_slice(&data_len.to_le_bytes());
                    for v in &tensor.data {
                        serialized.extend_from_slice(&v.to_le_bytes());
                    }
                }
            } else {
                panic!("Layer in Sequential does not implement SerializableLayer");
            }
        }

        serialized
    }

    /// Load sequential from serialized bytes
    pub fn deserialize(&mut self, data: &[u8], factory: &dyn Fn(&str) -> Box<dyn Layer>) {
        let mut pos = 0;
        self.layers.clear();

        while pos < data.len() {
            let type_len = data[pos] as usize;
            pos += 1;
            let layer_type = std::str::from_utf8(&data[pos..pos + type_len]).unwrap();
            pos += type_len;

            let cfg_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            let cfg_bytes = &data[pos..pos + cfg_len];
            pos += cfg_len;

            // Create layer using factory
            let layer = factory(layer_type);

            // Load config
            if let Some(sl) = layer.as_serializable() {
                sl.load_config(cfg_bytes);

                // load parameters
                let param_count = data[pos] as usize;
                pos += 1;
                let mut params = vec![];

                for _ in 0..param_count {
                    // Read name
                    let name_len = data[pos] as usize;
                    pos += 1;
                    let name = std::str::from_utf8(&data[pos..pos + name_len])
                        .unwrap()
                        .to_string();
                    pos += name_len;

                    // Read shape
                    let shape_len = data[pos] as usize;
                    pos += 1;
                    let mut shape = vec![];
                    for _ in 0..shape_len {
                        let dim =
                            u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
                        shape.push(dim);
                        pos += 4;
                    }

                    // Read tensor data
                    let data_len =
                        u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
                    pos += 4;
                    let mut tensor_data = vec![];
                    for _ in 0..data_len {
                        let val = f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                        tensor_data.push(val);
                        pos += 4;
                    }

                    params.push((
                        name,
                        Tensor {
                            data: tensor_data,
                            shape,
                            node: Some(Node::new(
                                vec![],
                                Box::new(crate::grad::leaf_grad::LeafGrad {}),
                            )),
                        },
                    ));
                }

                sl.load_parameters(params);
            }

            self.layers.push(layer);
        }
    }

    /// Save to disk
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) {
        std::fs::write(path, self.serialize()).expect("Failed to write sequential to disk");
    }

    /// Load from disk
    pub fn load<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
        factory: &dyn Fn(&str) -> Box<dyn Layer>,
    ) {
        let bytes = std::fs::read(path).expect("Failed to read file");
        self.deserialize(&bytes, factory);
    }
}
