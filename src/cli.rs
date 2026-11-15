//! Terminal front end (`seraph_engine --cli`): chat, single shot, WebSocket server.

use crate::engine::{self, Chat, Chunk, Engine};
use crate::server;
use std::io::{self, BufRead, Write};
use std::net::TcpListener;

fn prompt_user(label: &str) -> String {
    print!("{}", label);
    io::stdout().flush().unwrap();
    let mut line = String::new();
    io::stdin().lock().read_line(&mut line).unwrap();
    line.trim().to_string()
}

pub fn run() {
    println!("--- Seraph Engine ---");
    println!("Backend: 1) CUDA  2) CPU");
    let cuda = prompt_user("Choice (1-2): ") != "2";

    let models = engine::list_models();
    if models.is_empty() {
        eprintln!("No .gguf models found in {:?}", engine::models_dir());
        return;
    }
    println!("\nModels:");
    for (i, p) in models.iter().enumerate() {
        println!("  {}) {}", i + 1, p.file_name().unwrap().to_string_lossy());
    }
    let idx = prompt_user(&format!("Select (1-{}): ", models.len()))
        .parse::<usize>()
        .unwrap_or(1)
        .saturating_sub(1)
        .min(models.len() - 1);
    println!("Loading: {:?}", models[idx]);
    let (mut engine, info) = match Engine::load(&models[idx], cuda, |l| println!("{l}")) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("Load failed: {e}");
            return;
        }
    };
    println!("Context limit: {} tokens", info.ctx_limit);

    let system_prompt = engine::load_system_prompt();
    println!("System prompt loaded.\n");

    println!("Mode: 1) Chat  2) Single shot  3) WebSocket server");
    match prompt_user("Choice (1-3): ").as_str() {
        "2" => {
            println!("Single shot mode.\n");
            run_single_shot(&mut engine, &system_prompt);
        }
        "3" => {
            let port = prompt_user("Port (default 8543): ");
            let port = if port.is_empty() { "8543".to_string() } else { port };
            let addr = format!("127.0.0.1:{port}");
            let listener = match TcpListener::bind(&addr) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("Couldn't bind {addr}: {e}");
                    return;
                }
            };
            println!("WebSocket server listening on ws://{addr}");
            println!("One client at a time. Ctrl+C to stop.\n");
            server::serve(listener, Some((engine, info)), system_prompt);
        }
        _ => {
            println!("Chat mode. /reset to clear, /quit to exit.\n");
            run_chat(&mut engine, &system_prompt);
        }
    }
}

/// Prints visible reply text as it streams (reasoning is hidden).
fn print_chunk(c: Chunk) -> bool {
    if let Chunk::Text(t) = c {
        print!("{t}");
        io::stdout().flush().ok();
    }
    true
}

fn run_single_shot(engine: &mut Engine, system_prompt: &str) {
    loop {
        let input = prompt_user("You: ");
        if input.is_empty() || input == "/quit" || input == "/exit" {
            println!("Goodbye.");
            break;
        }
        let tokens = engine.template.single(&engine.tok, system_prompt, &input);
        print!("Seraph: ");
        let result = engine.generate(tokens, 2048, print_chunk);
        println!("\n");
        if let Err(e) = result {
            eprintln!("[error: {e}]\n");
        }
    }
}

fn run_chat(engine: &mut Engine, system_prompt: &str) {
    let mut chat = Chat::new(engine, system_prompt);
    loop {
        let input = prompt_user("You: ");
        if input.is_empty() || input == "/quit" || input == "/exit" {
            println!("Goodbye.");
            break;
        }
        if input == "/reset" {
            chat.reset(engine);
            println!("History cleared.\n");
            continue;
        }
        print!("Seraph: ");
        let result = chat.send(engine, &input, 2048, print_chunk);
        println!("\n");
        match result {
            Ok(stats) => {
                if stats.history_cleared {
                    println!("[context was full — earlier history was cleared]");
                }
                println!("[context: {} / {} tokens]", stats.ctx_len, engine.ctx_limit);
            }
            Err(e) => eprintln!("[error: {e}]\n"),
        }
    }
}
