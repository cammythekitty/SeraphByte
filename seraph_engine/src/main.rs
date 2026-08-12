use candle_core::{Device, Tensor};
use candle_core::quantized::gguf_file;
use candle_transformers::generation::LogitsProcessor;

use candle_transformers::models::quantized_llama;
use candle_transformers::models::quantized_qwen2;
use candle_transformers::models::quantized_qwen3;

use tokenizers::Tokenizer;
use std::fs::{self, File};
use std::io::{self, Write, Seek};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;

use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::protocol::Message;

use serde::Deserialize;

// Track the active model architecture on the GPU
enum ActiveModel {
    Llama(quantized_llama::ModelWeights),
    Qwen2(quantized_qwen2::ModelWeights),
    Qwen3(quantized_qwen3::ModelWeights),
}

// Incoming WebSocket frame from the frontend
// Either a plain text prompt (legacy) or a JSON object with config
#[derive(Deserialize, Debug)]
struct PromptFrame {
    prompt: String,
    #[serde(default = "default_temp")]
    temperature: f64,
    #[serde(default = "default_topp")]
    top_p: f64,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    system_prompt: Option<String>,
}

fn default_temp() -> f64 { 0.7 }
fn default_topp() -> f64 { 0.9 }
fn default_max_tokens() -> usize { 2048 }

// Commands passed from the WebSocket network stack into the heavy GPU thread
struct InferenceJob {
    frame: PromptFrame,
    fallback_system_prompt: String,
    tx_tokens: mpsc::Sender<String>,
}

// Resolve the tokenizer path relative to the binary, falling back to cwd
fn find_tokenizer_json() -> PathBuf {
    // Try next to the binary first
    if let Ok(exe) = std::env::current_exe() {
        let candidate = exe.parent().unwrap_or(Path::new(".")).join("tokenizer.json");
        if candidate.exists() { return candidate; }
    }
    // Try cwd
    let cwd_candidate = PathBuf::from("tokenizer.json");
    if cwd_candidate.exists() { return cwd_candidate; }
    // Give up with a clear error path so the message is meaningful
    PathBuf::from("tokenizer.json")
}

// Read the `general.architecture` key from GGUF metadata.
// Returns the arch string (e.g. "qwen3", "qwen2", "llama") or "llama" as fallback.
fn read_gguf_arch(content: &gguf_file::Content) -> String {
    content
        .metadata
        .get("general.architecture")
        .and_then(|v| {
            if let gguf_file::Value::String(s) = v {
                Some(s.clone())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "llama".to_string())
}

#[tokio::main]
async fn main() {
    println!("--- Initializing Auto-Detect GGUF Seraph Engine ---");

    // 1. Select hardware backend
    println!("Select inference backend:");
    println!("1) GPU (CUDA)");
    println!("2) CPU");

    print!("Choice (1-2): ");
    io::stdout().flush().unwrap();

    let mut backend = String::new();
    io::stdin()
        .read_line(&mut backend)
        .expect("Failed to read selection");

    let device = match backend.trim() {
        "2" => {
            println!("Using CPU execution.");
            Device::Cpu
        }
        _ => {
            match Device::new_cuda(0) {
                Ok(device) => {
                    println!("Using CUDA GPU.");
                    device
                }
                Err(_) => {
                    println!("CUDA initialization failed. Falling back to CPU.");
                    Device::Cpu
                }
            }
        }  
    };

println!("Engine target bound to hardware: {:?}", device);

    // 2. Resolve ~/Documents/Ai_Models
    let home_dir = dirs::home_dir().expect("Could not find the system home directory.");
    let models_dir = home_dir.join("Documents").join("Ai_Models");

    if !models_dir.exists() {
        fs::create_dir_all(&models_dir).expect("Failed to create Ai_Models directory.");
    }

    // 3. Scan for .gguf files
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
        return;
    }

    // 4. Display the model menu
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
    let file_name_str = selected_model_path
        .file_name().unwrap().to_string_lossy().to_string();
    println!("Loading weights from: {:?}", selected_model_path);

    // 5. Open the binary stream safely
    let mut file = File::open(selected_model_path).unwrap();
    let model_content = gguf_file::Content::read(&mut file)
        .expect("Failed to read GGUF container headers.");

    // 6. Read architecture from GGUF metadata (reliable) rather than filename sniffing
    let arch = read_gguf_arch(&model_content);
    println!("Detected GGUF architecture: {}", arch);

    // 7. Route to the correct loader based on the arch string
    let mut active_model = match arch.as_str() {
        "qwen3" => {
            println!("Configuring matrix layers for Qwen3 architecture layout...");
            let weights = quantized_qwen3::ModelWeights::from_gguf(model_content, &mut file, &device)
                .expect("Failed to build Qwen3 architecture layout.");
            ActiveModel::Qwen3(weights)
        }
        "qwen2" => {
            println!("Configuring matrix layers for Qwen2 architecture layout...");
            let weights = quantized_qwen2::ModelWeights::from_gguf(model_content, &mut file, &device)
                .expect("Failed to build Qwen2 architecture layout.");
            ActiveModel::Qwen2(weights)
        }
        _ => {
            // Covers "llama", "mistral", "mixtral", and anything else unknown
            println!("Configuring matrix layers for Llama/Mistral architecture layout...");
            let weights = quantized_llama::ModelWeights::from_gguf(model_content, &mut file, &device)
                .expect("Failed to build Llama/Mistral architecture layout.");
            ActiveModel::Llama(weights)
        }
    };

    // 8. Load tokenizer — search relative to binary, then cwd
    let tokenizer_path = find_tokenizer_json();
    println!("Loading tokenizer from: {:?}", tokenizer_path);
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .expect("Failed to load tokenizer.json — place it next to the binary or in the working directory.");

    // 9. Load default system prompt from markdown (used when frontend doesn't send one)
    let system_prompt_path = Path::new("system_prompt.md");
    if !system_prompt_path.exists() {
        let default_prompt = "# System Instructions\nYou are Seraph, a precise and advanced AI assistant.";
        fs::write(system_prompt_path, default_prompt).expect("Failed to create default system_prompt.md");
    }
    let default_system_prompt = fs::read_to_string(system_prompt_path)
        .expect("Failed to read system_prompt.md")
        .trim()
        .to_string();
    println!("Loaded system prompt configuration from system_prompt.md");

    // Broadcast the active model name to every new connection via this shared string
    let model_name = file_name_str.clone();

    // 10. Cross-thread channel: network → GPU
    let (tx_job, rx_job) = mpsc::channel::<InferenceJob>();

    // 11. Spawn the GPU worker thread
    thread::spawn(move || {
        println!("GPU Hardware Thread worker active and bound successfully.");

        while let Ok(job) = rx_job.recv() {
            println!("GPU executing incoming prompt request...");

            // Prefer system prompt from the frame; fall back to file-loaded default
            let sys = job.frame.system_prompt
                .as_deref()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or(&job.fallback_system_prompt);

            let formatted_prompt = format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                sys,
                job.frame.prompt
            );

            let encoding = tokenizer.encode(formatted_prompt, true).expect("Encoding failed");
            let mut tokens = encoding.get_ids().to_vec();

            // Build LogitsProcessor from per-request config
            let top_p = if job.frame.top_p > 0.0 && job.frame.top_p < 1.0 {
                Some(job.frame.top_p)
            } else {
                None
            };
            let mut logits_processor = LogitsProcessor::new(
                299792458,
                Some(job.frame.temperature),
                top_p,
            );

            let sample_len = job.frame.max_tokens;

            // Autoregressive token loop
            for i in 0..sample_len {
                let context_size = if i == 0 { tokens.len() } else { 1 };
                let start_pos = tokens.len() - context_size;

                let input_slice = &tokens[start_pos..];
                let input_tensor = Tensor::new(input_slice, &device)
                    .unwrap().unsqueeze(0).unwrap();

                let logits = match &mut active_model {
                    ActiveModel::Qwen3(m) => m.forward(&input_tensor, start_pos).unwrap(),
                    ActiveModel::Qwen2(m) => m.forward(&input_tensor, start_pos).unwrap(),
                    ActiveModel::Llama(m) => m.forward(&input_tensor, start_pos).unwrap(),
                };

                let logits = logits.squeeze(0).unwrap();
                let next_token = logits_processor.sample(&logits).unwrap();

                // EOS token IDs:
                // Qwen2 / Qwen3: 151645 = <|im_end|>, 151643 = <|endoftext|>
                // Llama / Mistral: 2 = </s>, 1 = <s> (used as stop in some quants)
                let is_eos = match &active_model {
                    ActiveModel::Qwen3(_) => next_token == 151645 || next_token == 151643,
                    ActiveModel::Qwen2(_) => next_token == 151645 || next_token == 151643,
                    ActiveModel::Llama(_) => next_token == 2 || next_token == 1,
                };
                if is_eos { break; }

                tokens.push(next_token);

                if let Ok(token_text) = tokenizer.decode(&[next_token], true) {
                    let clean_text = token_text
                        .replace('\u{2581}', " ")
                        .replace("<0x0A>", "\n");

                    if job.tx_tokens.send(clean_text).is_err() {
                        break; // Client disconnected
                    }
                }
            }
            println!("GPU Inference job complete.");
        }
    });

    // 12. WebSocket server
    let port = "8543";
    let addr = format!("127.0.0.1:{}", port);
    let listener = TcpListener::bind(&addr).await.expect("Failed to bind to safe address");
    println!("Engine Socket Gateway listening live on: ws://{}", addr);

    while let Ok((stream, _)) = listener.accept().await {
        let tx_job_clone = tx_job.clone();
        let default_sys_clone = default_system_prompt.clone();
        let model_name_clone = model_name.clone();

        tokio::spawn(async move {
            let ws_stream = accept_async(stream).await;
            if let Ok(mut ws) = ws_stream {
                println!("New system control socket handshake complete.");

                // Send the model name immediately on connect so the frontend can display it
                let hello = serde_json::json!({
                    "type": "model_info",
                    "model": model_name_clone
                });
                let _ = ws.send(Message::Text(hello.to_string())).await;

                while let Some(Ok(msg)) = ws.next().await {
                    if let Message::Text(raw) = msg {
                        // Try to parse as JSON PromptFrame; fall back to plain-text prompt
                        let frame: PromptFrame = if raw.trim_start().starts_with('{') {
                            match serde_json::from_str(&raw) {
                                Ok(f) => f,
                                Err(e) => {
                                    eprintln!("Failed to parse JSON frame: {e}");
                                    continue;
                                }
                            }
                        } else {
                            // Legacy plain-text: wrap with defaults
                            PromptFrame {
                                prompt: raw,
                                temperature: default_temp(),
                                top_p: default_topp(),
                                max_tokens: default_max_tokens(),
                                system_prompt: None,
                            }
                        };

                        let (tx_tokens, rx_tokens) = mpsc::channel::<String>();

                        let job = InferenceJob {
                            frame,
                            fallback_system_prompt: default_sys_clone.clone(),
                            tx_tokens,
                        };

                        if tx_job_clone.send(job).is_ok() {
                            while let Ok(token) = rx_tokens.recv() {
                                if ws.send(Message::Text(token)).await.is_err() {
                                    break;
                                }
                            }
                        }
                    }
                }
                println!("System control socket client disconnected.");
            }
        });
    }
}