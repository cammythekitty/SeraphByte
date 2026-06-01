use tch::{Tensor, nn, nn::Module};
use crate::model_traits::LanguageModel;

// --- Architecture 1: GPT-Style Model ---
pub struct SeraphGpt {
    pub embedding: nn::Embedding,
    pub projection: nn::Linear,
}

impl LanguageModel for SeraphGpt {
    fn new(vs: &nn::Path) -> Self {
        // Separate submodules using the / operator instead of a dot string
        let embedding = nn::embedding(vs / "transformer" / "wte", 50257, 768, Default::default());
        let projection = nn::linear(vs / "lm_head", 768, 50257, Default::default());
        
        SeraphGpt { embedding, projection }
    }

    fn forward(&self, input_ids: &Tensor) -> Tensor {
        let embedded_input = self.embedding.forward(input_ids);
        self.projection.forward(&embedded_input)
    }
}

// --- Architecture 2: Llama-Style Model ---
pub struct SeraphLlama {
    pub embedding: nn::Embedding,
    pub projection: nn::Linear,
}

impl LanguageModel for SeraphLlama {
    fn new(vs: &nn::Path) -> Self {
        let embedding = nn::embedding(vs / "transformer" / "wte", 50257, 768, Default::default());
        let projection = nn::linear(vs / "lm_head", 768, 50257, Default::default());
        SeraphLlama { embedding, projection }
    }

    fn forward(&self, input_ids: &Tensor) -> Tensor {
        let embedded_input = self.embedding.forward(input_ids);
        self.projection.forward(&embedded_input)
    }
}