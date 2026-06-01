// This file defines a standard contract that any model architecture must follow.

use tch::{Tensor, nn};

pub trait LanguageModel {
    /// Initializer that binds the model architecture to the loaded weights and device
    fn new(vs: &nn::Path) -> Self;
    /// Forward pass through the network layers
    fn forward(&self, input_ids: &Tensor) -> Tensor;
}