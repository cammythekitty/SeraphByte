use tch::Tensor;
use tokenizers::Tokenizer;
use crate::model_traits::LanguageModel;

pub fn generate<M: LanguageModel>(model: &M, tokenizer: &Tokenizer, prompt: &str) -> String {
    let encoding = tokenizer.encode(prompt, true).expect("Failed to encode prompt.");
    let mut current_tokens = encoding.get_ids().iter().map(|&id| id as i64).collect::<Vec<i64>>();
    
    // Set how many tokens maximum you want the LLM to generate per prompt
    let max_new_tokens = 12; 

    for _ in 0..max_new_tokens {
        // 1. Move the currently accumulated tokens onto the runtime device
        let input_tensor = Tensor::from_slice(&current_tokens).view([1, -1]);

        // 2. Run the forward pass through the architecture layers
        let logits = model.forward(&input_tensor);

        // 3. Isolate the final token prediction along the sequence dimension (-2)
        let last_token_logits = logits.select(1, -1);

        // 4. Extract the highest probability vocabulary token ID
        let next_token_id = last_token_logits.argmax(-1, false).int64_value(&[]);

        // Optional: Stop generating early if the model hits the End-Of-Text token boundary
        if next_token_id == 50256 {
            break;
        }

        // 5. Append the predicted token to our sequence array
        current_tokens.push(next_token_id);
    }

    // FIX: Cast the i64 tokens back to u32 for the tokenizer decoder layout
    let decoded_tokens: Vec<u32> = current_tokens.iter().map(|&id| id as u32).collect();

    // Decode the entire generated token chain back into a readable string
    tokenizer.decode(&decoded_tokens, true).unwrap()
}