# SeraphByte

A local LLM runner with a Rust backend and browser-based chat frontend.

## What it is

SeraphByte loads GGUF models (Llama/Mistral and Qwen/DeepSeek architectures) and serves them over a local WebSocket. The frontend is a plain HTML/CSS/JS chat interface that connects to it - no Electron, no Node server, just open `index.html` in a browser.

## Structure

```bash
seraph_engine/   — Rust backend (Axum, Tokio, Candle, tokenizers)
website/         — Frontend (index.html, app.js, theme.css)
seraph           — Launch script
```

## Requirements

- Rust toolchain (`cargo`)
- A GGUF model file placed in `~/Documents/Ai_Models/`
- A `tokenizer.json` placed next to the compiled binary (or in the working directory)
- CUDA-capable GPU recommended; falls back to CPU automatically

## Usage

```bash
# Build and run the engine
cd seraph_engine
cargo build --release
./target/release/seraph_engine
```

On startup it scans `~/Documents/Ai_Models/` and lets you pick a model. Once loaded, open `website/index.html` in your browser - it connects to `ws://127.0.0.1:8543` automatically.

Generation parameters (temperature, top-p, max tokens) and the system prompt are configurable from the Config panel in the sidebar and are sent to the backend with each message.

## Supported architectures

Model architecture is detected from the filename:

- filename contains `qwen` or `deepseek` → Qwen2 decoder
- everything else → Llama/Mistral decoder

## System prompt

On first run, a default `system_prompt.md` is created in the working directory. Edit it to change Seraph's default persona. You can also override it per-session from the Config panel in the UI.
