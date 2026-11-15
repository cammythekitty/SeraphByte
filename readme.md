# SeraphByte

A local LLM runner: a Rust inference engine (Candle) with a desktop interface, in one program.

## What it is

SeraphByte loads GGUF models (Llama/Mistral, Qwen2/DeepSeek-distill and Qwen3) and gives you a chat window for them. The interface is HTML/CSS/JS shown in a native window through the system webview. Its files are compiled into the binary, and the engine runs in the same process, so there is nothing else to start and nothing to open in a browser.

## Run

```bash
cargo run --release
```

or, once built, just `./target/release/seraph_engine`.

Pick a model under **Models** in the left column. The last model you used loads automatically next time. The **Load on CUDA / CPU** switch picks the device for the next load; CUDA falls back to CPU if it can't start.

`seraph_engine --cli` gives the terminal interface instead (chat, single shot, or a WebSocket server on a port you choose).

## Requirements

- Rust toolchain (`cargo`)
- GGUF model files in `~/Documents/Ai_Models/` (the tokenizer is read from the GGUF itself)
- Linux: WebKitGTK 4.1 and GTK 3 (`webkit2gtk-4.1` package)
- CUDA-capable GPU recommended (the build enables Candle's CUDA backend)

## Structure

```
src/
  main.rs       entry point: desktop app, or --cli
  app.rs        native window + webview, serves ui/ from the binary
  server.rs     WebSocket protocol between UI and engine
  engine.rs     model loading, generation, chat sessions
  template.rs   chat templates
  tokenizer.rs  GGUF tokenizer
  cli.rs        terminal interface
ui/             interface (index.html, app.js, theme.css, bundled fonts and libraries)
```

Changes in `ui/` take effect on the next `cargo build`.

## Notes

- Each message is answered on its own (system prompt + that one message). Conversations are kept for display in the app's storage (`~/.local/share/seraphbyte`), but the model doesn't see earlier turns yet.
- Reasoning from thinking models (`<think>…</think>`) is shown in a collapsible block above the answer.
- Temperature, top-p, max tokens and the system prompt are set in the right-hand panel and sent with each message. An empty system prompt uses `system_prompt.md`, which is created in the working directory on first run.
- Errors from the interface are printed to the terminal the app was started from.
