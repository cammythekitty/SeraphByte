//! WebSocket server used by the desktop app and by `--cli` mode 3.
//!
//! Server → client (JSON text frames unless noted):
//!   {"type":"models","models":[names],"dir":path,"loaded":name|null}  on connect and on "list"
//!   {"type":"model_info","model","arch","device","ctx_limit"}  on connect if loaded, after "load"
//!   {"type":"log","line"}                                  progress while loading
//!   raw token text (not JSON)                              reply text while generating
//!   {"type":"think","text"}                               reasoning inside <think>…</think>
//!   {"type":"done","tokens","prompt_tokens"}               end of a reply
//!   {"type":"error","message"}
//!
//! Client → server:
//!   {"type":"list"}
//!   {"type":"load","model":name,"cuda":bool}
//!   {prompt, temperature, top_p, max_tokens, system_prompt}   (or a plain-text prompt)
//!
//! One client at a time. Each prompt is stateless (system + one user turn).
//! A client stops generation by closing its socket; the next token send fails.

use crate::engine::{self, Chunk, Engine, ModelInfo, Sampling};
use serde_json::{json, Value};
use std::net::{TcpListener, TcpStream};
use tungstenite::{accept, Message, WebSocket};

type Ws = WebSocket<TcpStream>;

fn send_json(ws: &mut Ws, v: Value) -> bool {
    ws.send(Message::Text(v.to_string())).is_ok()
}

fn error(ws: &mut Ws, message: impl std::fmt::Display) -> bool {
    send_json(ws, json!({ "type": "error", "message": message.to_string() }))
}

fn model_info_json(info: &ModelInfo) -> Value {
    json!({
        "type": "model_info",
        "model": info.name,
        "arch": info.arch,
        "device": info.device,
        "ctx_limit": info.ctx_limit,
    })
}

fn models_json(loaded: Option<&(Engine, ModelInfo)>) -> Value {
    let names: Vec<String> = engine::list_models()
        .iter()
        .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .collect();
    json!({
        "type": "models",
        "models": names,
        "dir": engine::models_dir().to_string_lossy(),
        "loaded": loaded.map(|(_, i)| i.name.clone()),
    })
}

/// Serves clients forever. `loaded` is an engine that is already loaded, if any.
pub fn serve(listener: TcpListener, mut loaded: Option<(Engine, ModelInfo)>, default_system: String) {
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Connection error: {e}");
                continue;
            }
        };
        let mut ws = match accept(stream) {
            Ok(ws) => ws,
            Err(e) => {
                eprintln!("WS handshake failed: {e}");
                continue;
            }
        };
        println!("Client connected.");
        handle_client(&mut ws, &mut loaded, &default_system);
        println!("Client disconnected.");
    }
}

fn handle_client(ws: &mut Ws, loaded: &mut Option<(Engine, ModelInfo)>, default_system: &str) {
    if !send_json(ws, models_json(loaded.as_ref())) {
        return;
    }
    if let Some((_, info)) = loaded.as_ref() {
        if !send_json(ws, model_info_json(info)) {
            return;
        }
    }

    loop {
        let raw = match ws.read() {
            Ok(Message::Text(t)) => t,
            Ok(Message::Close(_)) | Err(_) => return,
            Ok(_) => continue,
        };

        let v: Option<Value> = if raw.trim_start().starts_with('{') {
            match serde_json::from_str(&raw) {
                Ok(v) => Some(v),
                Err(e) => {
                    if !error(ws, e) {
                        return;
                    }
                    continue;
                }
            }
        } else {
            None
        };

        let alive = match v.as_ref().and_then(|v| v["type"].as_str()) {
            Some("list") => send_json(ws, models_json(loaded.as_ref())),
            Some("load") => {
                let v = v.as_ref().unwrap();
                load(ws, loaded, v["model"].as_str().unwrap_or(""), v["cuda"].as_bool().unwrap_or(true))
            }
            _ => match loaded.as_mut() {
                Some((engine, _)) => generate(ws, engine, v.as_ref(), &raw, default_system),
                None => error(ws, "No model loaded. Choose one first."),
            },
        };
        if !alive {
            return;
        }
    }
}

/// Loads `name` from the models folder, replacing any loaded model.
fn load(ws: &mut Ws, loaded: &mut Option<(Engine, ModelInfo)>, name: &str, cuda: bool) -> bool {
    let Some(path) = engine::list_models()
        .into_iter()
        .find(|p| p.file_name().is_some_and(|n| n.to_string_lossy() == name))
    else {
        return error(ws, format!("Model not found: {name}"));
    };
    // Free the old weights (and VRAM) before loading new ones.
    *loaded = None;
    println!("Loading: {path:?}");
    let mut alive = true;
    let result = Engine::load(&path, cuda, |line| {
        println!("{line}");
        if alive {
            alive = send_json(ws, json!({ "type": "log", "line": line }));
        }
    });
    match result {
        Ok((engine, info)) => {
            println!("Loaded {} (context {})", info.name, info.ctx_limit);
            let ok = alive && send_json(ws, model_info_json(&info));
            *loaded = Some((engine, info));
            ok
        }
        Err(e) => {
            eprintln!("Load failed: {e}");
            alive && error(ws, format!("Load failed: {e}"))
        }
    }
}

fn generate(ws: &mut Ws, engine: &mut Engine, v: Option<&Value>, raw: &str, default_system: &str) -> bool {
    let defaults = Sampling::default();
    let (prompt, system, max_tokens, sampling) = match v {
        Some(v) => (
            v["prompt"].as_str().unwrap_or("").to_string(),
            v["system_prompt"].as_str().unwrap_or(default_system).to_string(),
            v["max_tokens"].as_u64().unwrap_or(2048) as usize,
            Sampling {
                temperature: v["temperature"].as_f64().unwrap_or(defaults.temperature).clamp(0.0, 2.0),
                top_p: v["top_p"].as_f64().unwrap_or(defaults.top_p).clamp(0.0, 1.0),
            },
        ),
        None => (raw.to_string(), default_system.to_string(), 2048, defaults),
    };
    if prompt.is_empty() {
        return true;
    }
    engine.sampling = sampling;
    let tokens = engine.template.single(&engine.tok, &system, &prompt);
    let prompt_len = tokens.len();
    let mut send_ok = true;
    let result = engine.generate(tokens, max_tokens, |c| {
        let msg = match c {
            Chunk::Text(text) => Message::Text(text.to_string()),
            Chunk::Think(text) => Message::Text(json!({ "type": "think", "text": text }).to_string()),
        };
        send_ok = ws.send(msg).is_ok();
        send_ok
    });
    if !send_ok {
        return false;
    }
    match result {
        Err(e) => {
            eprintln!("Generation error: {e}");
            error(ws, e)
        }
        Ok(all) => send_json(
            ws,
            json!({ "type": "done", "tokens": all.len() - prompt_len, "prompt_tokens": prompt_len }),
        ),
    }
}
