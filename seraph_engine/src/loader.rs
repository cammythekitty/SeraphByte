// This Loader is responsible for loading the model and tokenizer from disk and preparing them for inference.
// It will also handle any necessary preprocessing of the input text before it is passed to the generator.

use std::path::PathBuf;
use tch::{nn, Device, Tensor};
use tokenizers::Tokenizer;
pub struct Loader {
    pub model: nn::VarStore,
    pub tokenizer: Tokenizer,
}

impl Loader {
    pub fn new(model_path: PathBuf, tokenizer_path: PathBuf) -> Self {
        let device = Device::cuda_if_available();
        let mut model = nn::VarStore::new(device);
        // Load the model weights from the specified path
        model.load(model_path).expect("Failed to load model");
        // Load the tokenizer from the specified path
        let tokenizer = Tokenizer::from_file(tokenizer_path.to_str().unwrap()).expect("Failed to load tokenizer");
        Loader { model, tokenizer }
    }
}

impl Drop for Loader {
    fn drop(&mut self) {
        // Clean up resources if necessary
        println!("Loader is being dropped, cleaning up resources.");
    }
}

impl std::fmt::Debug for Loader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Loader")
            .field("model", &"nn::VarStore")
            .field("tokenizer", &"Tokenizer")
            .finish()
    }
}

impl std::fmt::Display for Loader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Loader with model and tokenizer")
    }
}  