//! Model loading, generation and chat sessions — shared by the CLI and the GUI.

use crate::template::Template;
use crate::tokenizer::{Tok, Utf8Stream};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama;
use crate::models::qwen2 as quantized_qwen2;
use crate::models::qwen3 as quantized_qwen3;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub const DEFAULT_SYSTEM_PROMPT: &str = "You are Seraph, a precise AI assistant.";
pub const SYSTEM_PROMPT_FILE: &str = "system_prompt.md";
/// Room kept free for the reply when deciding whether chat history still fits.
const REPLY_RESERVE: usize = 512;

enum ActiveModel {
    Llama(quantized_llama::ModelWeights),
    Qwen2(quantized_qwen2::ModelWeights),
    Qwen3(quantized_qwen3::ModelWeights),
}

impl ActiveModel {
    fn forward(&mut self, x: &Tensor, pos: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Llama(m) => m.forward(x, pos),
            Self::Qwen2(m) => m.forward(x, pos),
            Self::Qwen3(m) => m.forward(x, pos),
        }
    }

    fn clear_kv_cache(&mut self) {
        match self {
            Self::Llama(m) => m.clear_kv_cache(),
            Self::Qwen2(m) => m.clear_kv_cache(),
            Self::Qwen3(m) => m.clear_kv_cache(),
        }
    }
}

/// Facts about the loaded model, for display.
#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub name: String,
    pub arch: String,
    pub device: String,
    pub tokenizer: String,
    pub vocab_size: usize,
    pub template: String,
    pub ctx_limit: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Sampling {
    /// 0 = greedy.
    pub temperature: f64,
    /// 1 = disabled.
    pub top_p: f64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self { temperature: 0.7, top_p: 0.9 }
    }
}

/// A piece of streamed output.
pub enum Chunk<'a> {
    /// Visible reply text.
    Text(&'a str),
    /// Text inside a `<think>…</think>` block.
    Think(&'a str),
}

pub struct Engine {
    model: ActiveModel,
    pub tok: Tok,
    pub template: Template,
    device: Device,
    /// Max sequence length the model can address (prompt + generated tokens).
    pub ctx_limit: usize,
    /// Exactly the tokens currently held in the model's KV cache.
    kv_tokens: Vec<u32>,
    pub sampling: Sampling,
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

fn time_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(299792458)
}

/// `.gguf` files in `~/Documents/Ai_Models`, sorted by name.
pub fn models_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("Documents")
        .join("Ai_Models")
}

pub fn list_models() -> Vec<PathBuf> {
    let dir = models_dir();
    fs::create_dir_all(&dir).ok();
    let mut models: Vec<PathBuf> = fs::read_dir(&dir)
        .map(|rd| {
            rd.filter_map(|e| {
                let p = e.ok()?.path();
                (p.is_file() && p.extension()?.to_str()? == "gguf").then_some(p)
            })
            .collect()
        })
        .unwrap_or_default();
    models.sort();
    models
}

/// Reads `system_prompt.md`, creating it with the default prompt if missing.
pub fn load_system_prompt() -> String {
    let path = Path::new(SYSTEM_PROMPT_FILE);
    if !path.exists() {
        fs::write(path, DEFAULT_SYSTEM_PROMPT).ok();
    }
    fs::read_to_string(path)
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| DEFAULT_SYSTEM_PROMPT.to_string())
}

/// Opens CUDA device 0, or the CPU. Returns the device and a label for it.
pub fn make_device(cuda: bool) -> (Device, String) {
    if cuda {
        match Device::new_cuda(0) {
            Ok(d) => return (d, "CUDA".into()),
            Err(e) => return (Device::Cpu, format!("CPU (CUDA failed: {e})")),
        }
    }
    (Device::Cpu, "CPU".into())
}

impl Engine {
    /// Loads a GGUF model and its tokenizer. `log` receives progress lines.
    pub fn load(path: &Path, cuda: bool, mut log: impl FnMut(&str)) -> Result<(Engine, ModelInfo), String> {
        let (device, device_label) = make_device(cuda);
        log(&format!("Device: {device_label}"));

        let mut file = File::open(path).map_err(|e| format!("can't open model: {e}"))?;
        let mut content = gguf_file::Content::read(&mut file).map_err(|e| format!("failed to read GGUF: {e}"))?;

        let arch = match content.metadata.get("general.architecture") {
            Some(gguf_file::Value::String(s)) => s.clone(),
            _ => "llama".to_string(),
        };
        let meta_ctx = content
            .metadata
            .get(&format!("{arch}.context_length"))
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize);
        let chat_template = match content.metadata.get("tokenizer.chat_template") {
            Some(gguf_file::Value::String(s)) => Some(s.clone()),
            _ => None,
        };

        let tok = Tok::from_gguf(&mut content).map_err(|e| format!("tokenizer: {e}"))?;
        let template = Template::detect(chat_template.as_deref(), &tok);
        log(&format!(
            "Architecture: {arch}  Tokenizer: {} ({} tokens)  Template: {}",
            tok.kind(),
            tok.vocab_size(),
            template.name()
        ));

        log("Loading weights...");
        let (model, ctx_limit) = match arch.as_str() {
            "qwen3" => {
                let m = quantized_qwen3::ModelWeights::from_gguf(content, &mut file, &device)
                    .map_err(|e| format!("loading Qwen3 weights: {e}"))?;
                (ActiveModel::Qwen3(m), meta_ctx.unwrap_or(4096))
            }
            "qwen2" => {
                let m = quantized_qwen2::ModelWeights::from_gguf(content, &mut file, &device)
                    .map_err(|e| format!("loading Qwen2 weights: {e}"))?;
                (ActiveModel::Qwen2(m), meta_ctx.unwrap_or(4096))
            }
            _ => {
                let m = quantized_llama::ModelWeights::from_gguf(content, &mut file, &device)
                    .map_err(|e| format!("loading Llama weights: {e}"))?;
                // candle's llama RoPE table is fixed at MAX_SEQ_LEN positions.
                let limit = meta_ctx
                    .unwrap_or(quantized_llama::MAX_SEQ_LEN)
                    .min(quantized_llama::MAX_SEQ_LEN);
                (ActiveModel::Llama(m), limit)
            }
        };

        let info = ModelInfo {
            name: path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            arch,
            device: device_label,
            tokenizer: tok.kind().to_string(),
            vocab_size: tok.vocab_size(),
            template: template.name().to_string(),
            ctx_limit,
        };
        let engine = Engine {
            model,
            tok,
            template,
            device,
            ctx_limit,
            kv_tokens: Vec::new(),
            sampling: Sampling::default(),
        };
        Ok((engine, info))
    }

    /// Generates a reply to `tokens`, streaming output to `on_chunk`; returning
    /// `false` from it stops generation.
    ///
    /// If `tokens` extends what is already in the KV cache, only the new suffix is
    /// fed to the model. Returns `tokens` plus the generated ids (EOS excluded).
    pub fn generate<F>(&mut self, mut tokens: Vec<u32>, max_tokens: usize, mut on_chunk: F) -> Result<Vec<u32>, String>
    where
        F: FnMut(Chunk) -> bool,
    {
        if tokens.is_empty() {
            return Err("empty prompt".into());
        }
        if tokens.len() >= self.ctx_limit {
            return Err(format!(
                "prompt is {} tokens but the model's context limit is {}",
                tokens.len(),
                self.ctx_limit
            ));
        }
        let max_tokens = max_tokens.min(self.ctx_limit - tokens.len());

        let s = self.sampling;
        let temperature = (s.temperature > 0.0).then_some(s.temperature);
        let top_p = (s.top_p > 0.0 && s.top_p < 1.0).then_some(s.top_p);
        let mut sampler = LogitsProcessor::new(time_seed(), temperature, top_p);

        // Reuse the cache only when the new prompt strictly extends it; candle's
        // caches can't be truncated, so anything else starts from scratch. (Qwen3's
        // cache doesn't reset itself at position 0, so always clear explicitly.)
        let reuse = tokens.len() > self.kv_tokens.len() && tokens.starts_with(&self.kv_tokens);
        if !reuse {
            self.model.clear_kv_cache();
            self.kv_tokens.clear();
        }

        let mut utf8 = Utf8Stream::default();
        let mut buf = String::new();
        let mut in_think = false;
        // Trim leading whitespace at the start of the reply and of the reasoning.
        let mut text_started = false;
        let mut think_started = false;

        let mut emit = |text: &str, think: bool, text_started: &mut bool, think_started: &mut bool| -> bool {
            let started = if think { think_started } else { text_started };
            let text = if *started { text } else { text.trim_start() };
            if text.is_empty() {
                return true;
            }
            *started = true;
            on_chunk(if think { Chunk::Think(text) } else { Chunk::Text(text) })
        };

        for _ in 0..max_tokens {
            let fed = self.kv_tokens.len();
            let logits = (|| {
                let input = Tensor::new(&tokens[fed..], &self.device)?.unsqueeze(0)?;
                self.model.forward(&input, fed)?.squeeze(0)
            })();
            let logits = match logits {
                Ok(l) => l,
                Err(e) => {
                    // Cache state is unknown after a failed forward; drop it.
                    self.model.clear_kv_cache();
                    self.kv_tokens.clear();
                    return Err(format!("forward pass failed: {e}"));
                }
            };
            self.kv_tokens.extend_from_slice(&tokens[fed..]);

            let next = sampler.sample(&logits).map_err(|e| format!("sampling failed: {e}"))?;
            if self.tok.eos_ids.contains(&next) {
                break;
            }
            tokens.push(next);
            utf8.push(self.tok.token_bytes(next), &mut buf);

            // Split <think>...</think> from the reply, holding back a tag-sized tail
            // in case a tag is split across tokens.
            loop {
                let tag = if in_think { THINK_CLOSE } else { THINK_OPEN };
                if let Some(pos) = buf.find(tag) {
                    let before = buf[..pos].to_string();
                    buf.drain(..pos + tag.len());
                    let ok = emit(&before, in_think, &mut text_started, &mut think_started);
                    in_think = !in_think;
                    if !ok {
                        return Ok(tokens);
                    }
                } else {
                    let keep = tail_start(&buf, tag.len());
                    let out: String = buf.drain(..keep).collect();
                    if !emit(&out, in_think, &mut text_started, &mut think_started) {
                        return Ok(tokens);
                    }
                    break;
                }
            }
        }
        utf8.finish(&mut buf);
        if !buf.is_empty() {
            emit(&buf, in_think, &mut text_started, &mut think_started);
        }
        Ok(tokens)
    }
}

/// Byte index where the last `n - 1` chars of `s` begin (the part that could still be
/// the start of an `n`-char tag).
fn tail_start(s: &str, n: usize) -> usize {
    s.char_indices()
        .rev()
        .nth(n.saturating_sub(2))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Result of one chat turn.
#[derive(Clone, Copy, Debug, Default)]
pub struct TurnStats {
    pub generated: usize,
    pub secs: f64,
    pub ctx_len: usize,
    /// History was dropped because it no longer fit in the context window.
    pub history_cleared: bool,
}

impl TurnStats {
    pub fn tokens_per_sec(&self) -> f64 {
        if self.secs > 0.0 { self.generated as f64 / self.secs } else { 0.0 }
    }
}

/// A multi-turn conversation that keeps its token history (and so the KV cache).
pub struct Chat {
    ctx: Vec<u32>,
    first_turn: bool,
    system: String,
}

impl Chat {
    pub fn new(engine: &Engine, system: &str) -> Self {
        Self {
            ctx: engine.template.start(&engine.tok, system),
            first_turn: true,
            system: system.to_string(),
        }
    }

    pub fn reset(&mut self, engine: &Engine) {
        *self = Self::new(engine, &self.system.clone());
    }

    /// Changing the system prompt starts a new conversation.
    pub fn set_system(&mut self, engine: &Engine, system: &str) {
        if self.system != system {
            *self = Self::new(engine, system);
        }
    }

    pub fn send<F>(&mut self, engine: &mut Engine, user: &str, max_tokens: usize, on_chunk: F) -> Result<TurnStats, String>
    where
        F: FnMut(Chunk) -> bool,
    {
        let tpl = engine.template;
        let mut turn = tpl.user_turn(&engine.tok, &self.system, user, self.first_turn);
        let mut history_cleared = false;
        if !self.first_turn && self.ctx.len() + turn.len() + REPLY_RESERVE > engine.ctx_limit {
            self.reset(engine);
            history_cleared = true;
            turn = tpl.user_turn(&engine.tok, &self.system, user, true);
        }
        let mut ids = self.ctx.clone();
        ids.extend(turn);
        let prompt_len = ids.len();

        let start = Instant::now();
        let out = engine.generate(ids, max_tokens, on_chunk)?;
        let secs = start.elapsed().as_secs_f64();

        let generated = out.len() - prompt_len;
        self.ctx = out;
        self.ctx.extend(tpl.end_turn(&engine.tok));
        self.first_turn = false;
        Ok(TurnStats { generated, secs, ctx_len: self.ctx.len(), history_cleared })
    }
}
