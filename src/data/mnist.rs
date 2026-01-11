use std::fs::File;
use std::io::{Read, Result};
use std::path::PathBuf;
use crate::data::utils::one_hot;
use crate::Tensor;

#[derive(Debug)]
pub struct Mnist {
    pub images: Vec<Vec<f32>>, // flattened, normalized
    pub labels: Vec<u8>,
    pub rows: usize,
    pub cols: usize,
}

/// Read u32 stored in big endian
fn read_u32_be(bytes: &[u8]) -> u32 {
    u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

/// Load MNIST images
fn load_images(path: PathBuf) -> Result<(Vec<Vec<f32>>, usize, usize)> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let magic = read_u32_be(&buffer[0..4]);
    assert_eq!(magic, 2051, "Invalid MNIST image file");

    let num_images = read_u32_be(&buffer[4..8]) as usize;
    let rows = read_u32_be(&buffer[8..12]) as usize;
    let cols = read_u32_be(&buffer[12..16]) as usize;

    let image_size = rows * cols;
    let mut images = Vec::with_capacity(num_images);

    let mut offset = 16;
    for _ in 0..num_images {
        let img = buffer[offset..offset + image_size]
            .iter()
            .map(|&b| b as f32 / 255.0) // normalize
            .collect();
        images.push(img);
        offset += image_size;
    }

    Ok((images, rows, cols))
}

/// Load MNIST labels
fn load_labels(path: PathBuf) -> Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let magic = read_u32_be(&buffer[0..4]);
    assert_eq!(magic, 2049, "Invalid MNIST label file");

    let num_labels = read_u32_be(&buffer[4..8]) as usize;

    Ok(buffer[8..8 + num_labels].to_vec())
}

impl Mnist {
    /// Load the MNIST dataset
    pub fn load(image_path: PathBuf, label_path: PathBuf) -> Result<Self> {
        let (images, rows, cols) = load_images(image_path)?;
        let labels = load_labels(label_path)?;

        assert_eq!(images.len(), labels.len());

        Ok(Self {
            images,
            labels,
            rows,
            cols,
        })
    }
}

pub fn load_mnist_as_tensors(
    image_path: PathBuf,
    label_path: PathBuf,
    num_classes: usize,
) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
    let mnist = Mnist::load(image_path, label_path)?;

    let num_samples = mnist.images.len();
    let input_size = mnist.rows * mnist.cols;

    let mut inputs = Vec::with_capacity(num_samples);
    let mut targets = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        // Input tensor: [784, 1]
        let x = Tensor::from_vec_leaf(
            mnist.images[i].clone(),
            vec![input_size, 1],
        );

        // Target tensor: [num_classes, 1]
        let y = Tensor::from_vec_leaf(
            one_hot(mnist.labels[i], num_classes),
            vec![num_classes, 1],
        );

        inputs.push(x);
        targets.push(y);
    }

    Ok((inputs, targets))
}

