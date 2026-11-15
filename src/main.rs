mod app;
mod cli;
mod engine;
mod models;
mod server;
mod template;
mod tokenizer;

fn main() {
    if std::env::args().any(|a| a == "--cli") {
        cli::run();
    } else if let Err(e) = app::run() {
        eprintln!("Couldn't start SeraphByte: {e}");
        eprintln!("Run with --cli for the terminal interface.");
        std::process::exit(1);
    }
}
