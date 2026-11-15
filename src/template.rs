//! Chat prompt formats, picked from the GGUF's `tokenizer.chat_template`.
//!
//! A conversation is laid out as:
//!   start(system)  user_turn(..)  <reply>  end_turn()  user_turn(..)  <reply> ...
//! so chat mode can keep appending to the same token sequence (and KV cache).

use crate::tokenizer::Tok;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Template {
    /// `<|im_start|>role\n...<|im_end|>` — Qwen and most fine-tunes.
    ChatMl,
    /// `<|start_header_id|>role<|end_header_id|>\n\n...<|eot_id|>`
    Llama3,
    /// `<｜User｜>...<｜Assistant｜>` — DeepSeek-R1 and its distills.
    DeepSeek,
    /// `[INST] ... [/INST]` — Mistral / Llama 2 (system folded into first user turn).
    Inst,
}

impl Template {
    pub fn detect(chat_template: Option<&str>, tok: &Tok) -> Self {
        if let Some(t) = chat_template {
            if t.contains("<｜User｜>") || t.contains("<｜Assistant｜>") {
                return Self::DeepSeek;
            }
            if t.contains("<|start_header_id|>") {
                return Self::Llama3;
            }
            if t.contains("<|im_start|>") {
                return Self::ChatMl;
            }
            if t.contains("[INST]") {
                return Self::Inst;
            }
        }
        if tok.has_special("<|im_start|>") {
            Self::ChatMl
        } else if tok.has_special("<|start_header_id|>") {
            Self::Llama3
        } else if tok.has_special("<｜User｜>") {
            Self::DeepSeek
        } else {
            Self::Inst
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::ChatMl => "ChatML",
            Self::Llama3 => "Llama 3",
            Self::DeepSeek => "DeepSeek",
            Self::Inst => "[INST]",
        }
    }

    /// Start of a conversation: BOS plus the system prompt where the format has one.
    pub fn start(self, tok: &Tok, system: &str) -> Vec<u32> {
        let mut ids = tok.bos();
        match self {
            Self::ChatMl => {
                ids.extend(tok.special("<|im_start|>"));
                ids.extend(tok.encode(&format!("system\n{system}")));
                ids.extend(tok.special("<|im_end|>"));
                ids.extend(tok.encode("\n"));
            }
            Self::Llama3 => {
                ids.extend(tok.special("<|start_header_id|>"));
                ids.extend(tok.encode("system"));
                ids.extend(tok.special("<|end_header_id|>"));
                ids.extend(tok.encode(&format!("\n\n{system}")));
                ids.extend(tok.special("<|eot_id|>"));
            }
            Self::DeepSeek => ids.extend(tok.encode(system)),
            Self::Inst => {} // folded into the first user turn
        }
        ids
    }

    /// A user message followed by the assistant generation prompt.
    /// `first_turn` lets formats without a system role prepend it to the first message.
    pub fn user_turn(self, tok: &Tok, system: &str, user: &str, first_turn: bool) -> Vec<u32> {
        let mut ids = Vec::new();
        match self {
            Self::ChatMl => {
                ids.extend(tok.special("<|im_start|>"));
                ids.extend(tok.encode(&format!("user\n{user}")));
                ids.extend(tok.special("<|im_end|>"));
                ids.extend(tok.encode("\n"));
                ids.extend(tok.special("<|im_start|>"));
                ids.extend(tok.encode("assistant\n"));
            }
            Self::Llama3 => {
                ids.extend(tok.special("<|start_header_id|>"));
                ids.extend(tok.encode("user"));
                ids.extend(tok.special("<|end_header_id|>"));
                ids.extend(tok.encode(&format!("\n\n{user}")));
                ids.extend(tok.special("<|eot_id|>"));
                ids.extend(tok.special("<|start_header_id|>"));
                ids.extend(tok.encode("assistant"));
                ids.extend(tok.special("<|end_header_id|>"));
                ids.extend(tok.encode("\n\n"));
            }
            Self::DeepSeek => {
                ids.extend(tok.special("<｜User｜>"));
                ids.extend(tok.encode(user));
                ids.extend(tok.special("<｜Assistant｜>"));
            }
            Self::Inst => {
                let text = if first_turn && !system.is_empty() {
                    format!("[INST] {system}\n\n{user} [/INST]")
                } else {
                    format!("[INST] {user} [/INST]")
                };
                ids.extend(tok.encode(&text));
            }
        }
        ids
    }

    /// Closes an assistant reply so the next user turn can follow.
    pub fn end_turn(self, tok: &Tok) -> Vec<u32> {
        match self {
            Self::ChatMl => {
                let mut ids = tok.special("<|im_end|>");
                ids.extend(tok.encode("\n"));
                ids
            }
            Self::Llama3 => tok.special("<|eot_id|>"),
            Self::DeepSeek => tok.special("<｜end▁of▁sentence｜>"),
            Self::Inst => tok.eos_token().into_iter().collect(),
        }
    }

    /// A complete single-turn prompt.
    pub fn single(self, tok: &Tok, system: &str, user: &str) -> Vec<u32> {
        let mut ids = self.start(tok, system);
        ids.extend(self.user_turn(tok, system, user, true));
        ids
    }
}
