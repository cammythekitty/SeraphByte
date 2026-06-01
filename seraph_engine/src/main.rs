use candle_core::{Device, Tensor};
use candle_core::quantized::gguf_file;
use candle_transformers::generation::LogitsProcessor;

// Import the verified layout decoders
use candle_transformers::models::quantized_llama;
use candle_transformers::models::quantized_qwen2;

use tokenizers::Tokenizer;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

// Track the active model architecture on the GPU
enum ActiveModel {
    Llama(quantized_llama::ModelWeights), // Handles both Llama and Mistral seamlessly
    Qwen(quantized_qwen2::ModelWeights),   // Handles Qwen and DeepSeek-Distills
}

fn main() {
    println!("--- Initializing Auto-Detect GGUF Seraph Engine ---");

    // 1. Setup GPU Acceleration
    let device = Device::new_cuda(0).unwrap_or_else(|_| {
        println!("CUDA GPU initialization failed. Falling back to CPU execution.");
        Device::Cpu
    });
    println!("Engine target bound to hardware: {:?}", device);

    // 2. Resolve the ~/Documents/Ai_Models directory path dynamically
    let home_dir = dirs::home_dir().expect("Could not find the system home directory.");
    let models_dir = home_dir.join("Documents").join("Ai_Models");

    if !models_dir.exists() {
        fs::create_dir_all(&models_dir).expect("Failed to create Ai_Models directory.");
    }

    // 3. Scan the folder and collect all .gguf files
    let gguf_models: Vec<PathBuf> = fs::read_dir(&models_dir)
        .expect("Failed to read models directory")
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.is_file() && path.extension()?.to_str()? == "gguf" {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    if gguf_models.is_empty() {
        println!("\nError: No .gguf models found in {:?}", models_dir);
        println!("Please drop your .gguf files into that directory and restart.");
        return;
    }

    // 4. Display the dynamically generated menu
    println!("\nDetected Models:");
    for (idx, path) in gguf_models.iter().enumerate() {
        if let Some(file_name) = path.file_name() {
            println!("{}) {}", idx + 1, file_name.to_string_lossy());
        }
    }

    print!("Select a model to boot (1-{}): ", gguf_models.len());
    io::stdout().flush().unwrap();

    let mut choice = String::new();
    io::stdin().read_line(&mut choice).expect("Failed to read selection");
    
    let selection_idx: usize = match choice.trim().parse::<usize>() {
        Ok(num) if num > 0 && num <= gguf_models.len() => num - 1,
        _ => 0,
    };

    let selected_model_path = &gguf_models[selection_idx];
    let file_name_lower = selected_model_path.file_name().unwrap().to_string_lossy().to_lowercase();
    println!("Loading weights from: {:?}", selected_model_path);

    // 5. Open the binary stream safely
    let mut file = File::open(selected_model_path).unwrap();
    let model_content = gguf_file::Content::read(&mut file)
        .expect("Failed to read GGUF container headers.");
    
    // 6. Direct architectures to their corresponding tensor math runners
    let mut active_model = if file_name_lower.contains("qwen") || file_name_lower.contains("deepseek") {
        println!("Configuring matrix layers for Qwen/DeepSeek architecture layout...");
        let weights = quantized_qwen2::ModelWeights::from_gguf(model_content, &mut file, &device)
            .expect("Failed to build Qwen architecture layout.");
        ActiveModel::Qwen(weights)
    } else {
        // This structural loader natively processes both Mistral and Llama architecture profiles
        println!("Configuring matrix layers for Llama/Mistral architecture layout...");
        let weights = quantized_llama::ModelWeights::from_gguf(model_content, &mut file, &device)
            .expect("Failed to build Llama/Mistral architecture layout.");
        ActiveModel::Llama(weights)
    };

    // 7. Load Companion Tokenizer Layout
    let tokenizer = Tokenizer::from_file("tokenizer.json")
        .expect("Failed to load tokenizer.json");

    // NEW: Dynamically load the system prompt from a local Markdown file (.md)
    let system_prompt_path = Path::new("system_prompt.md");
    if !system_prompt_path.exists() {
        let default_prompt = "# System Instructions\nYou are Seraph, a precise and advanced AI assistant.";
        fs::write(system_prompt_path, default_prompt)
            .expect("Failed to create default system_prompt.md");
    }
    let system_prompt = fs::read_to_string(system_prompt_path)
        .expect("Failed to read system_prompt.md")
        .trim()
        .to_string();

    println!("Loaded system prompt configuration from system_prompt.md");
    println!("GGUF Model loaded and initialized into GPU context. Ready for real text generation!");

    // 8. Generation configs
    let sample_len = 25000; // High token limit for long generations
    let mut logits_processor = LogitsProcessor::new(299792458, Some(0.2), None);

    // 9. Interactive Prompt Processing Loop
    loop {
        print!("\nEnter Prompt: ");
        io::stdout().flush().unwrap();

        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input).expect("Failed to read line");
        
        let trimmed_input = user_input.trim();
        if trimmed_input.eq_ignore_ascii_case("exit") {
            println!("Shutting down engine gracefully.");
            break;
        }
        if trimmed_input.is_empty() {
            continue;
        }

        // Apply ChatML formatting utilizing the prompt loaded from the markdown file
        let formatted_templated_prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            system_prompt,
            trimmed_input
        );

        let encoding = tokenizer.encode(formatted_templated_prompt, true).expect("Encoding failed");
        let mut tokens = encoding.get_ids().to_vec();

        print!("Generated Text: ");
        io::stdout().flush().unwrap();

        // Autoregressive token generation loop
        for i in 0..sample_len {
            let context_size = if i == 0 { tokens.len() } else { 1 };
            let start_pos = tokens.len() - context_size;
            
            let input_slice = &tokens[start_pos..];
            let input_tensor = Tensor::new(input_slice, &device).unwrap().unsqueeze(0).unwrap();
            
            // Route execution seamlessly depending on the variant chosen
            let logits = match &mut active_model {
                ActiveModel::Qwen(m) => m.forward(&input_tensor, start_pos).unwrap(),
                ActiveModel::Llama(m) => m.forward(&input_tensor, start_pos).unwrap(),
            };
            
            let logits = logits.squeeze(0).unwrap();
            let next_token = logits_processor.sample(&logits).unwrap();
            
            // Stop early if the model hits an End-Of-Sequence token (Qwen: 151645, Llama/Mistral: 2)
            if next_token == 151645 || next_token == 2 {
                break;
            }
            
            tokens.push(next_token);

            if let Ok(token_text) = tokenizer.decode(&[next_token], true) {
                let clean_text = token_text.replace(" ", " ").replace("<0x0A>", "\n");
                print!("{}", clean_text);
                io::stdout().flush().unwrap();
            }
        }
        println!();
    }
}