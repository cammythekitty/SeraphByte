//! Tokenization built from GGUF metadata.
//!
//! * Byte-level BPE vocabularies (`tokenizer.ggml.model == "gpt2"`: Qwen, Llama 3,
//!   DeepSeek distills) go through the HF `tokenizers` crate via candle's GGUF bridge,
//!   so encoding matches the reference tokenizer (merges + pre-tokenizer regex).
//! * SentencePiece vocabularies (`"llama"`: Llama 2, Mistral) use a score-driven BPE
//!   that mirrors llama.cpp's SPM tokenizer, with `<0xNN>` byte fallback.
//!
//! User text is always encoded as plain text: special tokens are only ever inserted
//! by id, so typing `<|im_end|>` in a prompt cannot break the chat template.

use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::tokenizer::TokenizerFromGguf;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

// llama.cpp token types
const TYPE_NORMAL: i64 = 1;
const TYPE_CONTROL: i64 = 3;
const TYPE_USER_DEFINED: i64 = 4;
const TYPE_BYTE: i64 = 6;

pub struct Tok {
    encoder: Encoder,
    token_map: HashMap<String, u32>,
    token_types: Vec<i64>,
    /// Raw bytes each token decodes to (empty for control tokens).
    token_bytes: Vec<Vec<u8>>,
    pub eos_ids: HashSet<u32>,
    bos_id: Option<u32>,
    add_bos: bool,
}

enum Encoder {
    Hf(Box<tokenizers::Tokenizer>),
    Spm(Spm),
}

fn gguf_int(v: &Value) -> Option<i64> {
    Some(match v {
        Value::U8(x) => *x as i64,
        Value::I8(x) => *x as i64,
        Value::U16(x) => *x as i64,
        Value::I16(x) => *x as i64,
        Value::U32(x) => *x as i64,
        Value::I32(x) => *x as i64,
        Value::U64(x) => *x as i64,
        Value::I64(x) => *x,
        _ => return None,
    })
}

fn meta_str<'a>(c: &'a gguf_file::Content, key: &str) -> Option<&'a str> {
    match c.metadata.get(key) {
        Some(Value::String(s)) => Some(s.as_str()),
        _ => None,
    }
}

fn meta_bool(c: &gguf_file::Content, key: &str) -> Option<bool> {
    match c.metadata.get(key) {
        Some(Value::Bool(b)) => Some(*b),
        _ => None,
    }
}

fn meta_u32(c: &gguf_file::Content, key: &str) -> Option<u32> {
    c.metadata.get(key).and_then(gguf_int).map(|x| x as u32)
}

fn meta_array<'a>(c: &'a gguf_file::Content, key: &str) -> &'a [Value] {
    match c.metadata.get(key) {
        Some(Value::Array(a)) => a.as_slice(),
        _ => &[],
    }
}

/// GPT-2's byte -> printable-unicode table, inverted (char -> byte).
fn gpt2_unicode_to_byte() -> HashMap<char, u8> {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32)
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.into_iter()
        .zip(cs)
        .map(|(b, c)| (char::from_u32(c).unwrap(), b as u8))
        .collect()
}

fn parse_byte_token(s: &str) -> Option<u8> {
    let hex = s.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

impl Tok {
    /// Builds the tokenizer. Takes `&mut` because the pre-tokenizer name is normalised
    /// in the metadata before handing it to candle (model loading never reads it).
    pub fn from_gguf(content: &mut gguf_file::Content) -> Result<Self, String> {
        let tokens: Vec<String> = meta_array(content, "tokenizer.ggml.tokens")
            .iter()
            .map(|v| match v {
                Value::String(s) => s.clone(),
                _ => String::new(),
            })
            .collect();
        if tokens.is_empty() {
            return Err("GGUF has no tokenizer.ggml.tokens".into());
        }
        let mut token_types: Vec<i64> = meta_array(content, "tokenizer.ggml.token_type")
            .iter()
            .map(|v| gguf_int(v).unwrap_or(TYPE_NORMAL))
            .collect();
        token_types.resize(tokens.len(), TYPE_NORMAL);

        let model_kind = meta_str(content, "tokenizer.ggml.model")
            .unwrap_or("llama")
            .to_lowercase();
        let is_spm = model_kind == "llama";

        let token_map: HashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();

        // Decoding table.
        let byte_map = gpt2_unicode_to_byte();
        let token_bytes: Vec<Vec<u8>> = tokens
            .iter()
            .zip(&token_types)
            .map(|(t, &ty)| match ty {
                TYPE_CONTROL => Vec::new(),
                // Added tokens such as <think> are stored as literal text.
                TYPE_USER_DEFINED => t.as_bytes().to_vec(),
                _ if is_spm => match parse_byte_token(t) {
                    Some(b) if ty == TYPE_BYTE => vec![b],
                    _ => t.replace('\u{2581}', " ").into_bytes(),
                },
                _ => {
                    let mut out = Vec::with_capacity(t.len());
                    for ch in t.chars() {
                        match byte_map.get(&ch) {
                            Some(&b) => out.push(b),
                            None => {
                                let mut buf = [0u8; 4];
                                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                            }
                        }
                    }
                    out
                }
            })
            .collect();

        // End-of-generation tokens: only control tokens count, so an ordinary vocab
        // entry that happens to read "</s>" (Qwen has one) doesn't stop generation.
        let mut eos_ids = HashSet::new();
        if let Some(id) = meta_u32(content, "tokenizer.ggml.eos_token_id") {
            eos_ids.insert(id);
        }
        for s in [
            "<|im_end|>", "<|endoftext|>", "</s>", "<|end_of_sentence|>",
            "<｜end▁of▁sentence｜>", "<|eot_id|>", "<|eom_id|>", "<|end|>", "[EOS]",
        ] {
            if let Some(&id) = token_map.get(s) {
                if token_types[id as usize] == TYPE_CONTROL {
                    eos_ids.insert(id);
                }
            }
        }

        let bos_id = meta_u32(content, "tokenizer.ggml.bos_token_id");
        let add_bos = meta_bool(content, "tokenizer.ggml.add_bos_token").unwrap_or(is_spm);

        let encoder = if is_spm {
            let scores: Vec<f32> = meta_array(content, "tokenizer.ggml.scores")
                .iter()
                .map(|v| if let Value::F32(f) = v { *f } else { 0.0 })
                .collect();
            Encoder::Spm(Spm::new(
                &tokens,
                &token_types,
                &scores,
                meta_u32(content, "tokenizer.ggml.unknown_token_id"),
                meta_bool(content, "tokenizer.ggml.add_space_prefix").unwrap_or(true),
            ))
        } else if model_kind == "gpt2" {
            // candle only knows a few pre-tokenizer names and falls back to GPT-2
            // defaults (which add a prefix space) for the rest. Map onto the two regex
            // families it implements.
            let pre = meta_str(content, "tokenizer.ggml.pre").unwrap_or("").to_lowercase();
            let family = if pre.contains("qwen") { "qwen2" } else { "llama3" };
            content
                .metadata
                .insert("tokenizer.ggml.pre".into(), Value::String(family.into()));
            let mut hf = tokenizers::Tokenizer::from_gguf(content)
                .map_err(|e| format!("building tokenizer: {e}"))?;
            // Treat special-token strings in user text as plain text.
            hf.set_encode_special_tokens(true);
            Encoder::Hf(Box::new(hf))
        } else {
            return Err(format!("unsupported tokenizer model `{model_kind}`"));
        };

        Ok(Self { encoder, token_map, token_types, token_bytes, eos_ids, bos_id, add_bos })
    }

    pub fn kind(&self) -> &'static str {
        match self.encoder {
            Encoder::Hf(_) => "byte-level BPE",
            Encoder::Spm(_) => "SentencePiece",
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.token_bytes.len()
    }

    /// Encodes plain text (never produces special tokens, never adds BOS).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        match &self.encoder {
            Encoder::Hf(t) => t
                .encode(text, false)
                .map(|e| e.get_ids().to_vec())
                .unwrap_or_else(|e| {
                    eprintln!("[tokenizer error: {e}]");
                    Vec::new()
                }),
            Encoder::Spm(s) => s.encode(text),
        }
    }

    /// Id of a special (control / user-defined) token, if the vocab has it.
    pub fn special_id(&self, s: &str) -> Option<u32> {
        let &id = self.token_map.get(s)?;
        matches!(self.token_types[id as usize], TYPE_CONTROL | TYPE_USER_DEFINED).then_some(id)
    }

    pub fn has_special(&self, s: &str) -> bool {
        self.special_id(s).is_some()
    }

    /// A special token by id, falling back to encoding its text.
    pub fn special(&self, s: &str) -> Vec<u32> {
        match self.special_id(s) {
            Some(id) => vec![id],
            None => self.encode(s),
        }
    }

    /// BOS token if the model expects one at the start of a sequence.
    pub fn bos(&self) -> Vec<u32> {
        match (self.add_bos, self.bos_id) {
            (true, Some(id)) => vec![id],
            _ => Vec::new(),
        }
    }

    /// The first EOS id (used to close assistant turns in templates that need it).
    pub fn eos_token(&self) -> Option<u32> {
        self.eos_ids.iter().min().copied()
    }

    pub fn token_bytes(&self, id: u32) -> &[u8] {
        self.token_bytes.get(id as usize).map(Vec::as_slice).unwrap_or(&[])
    }
}

// ── SentencePiece BPE (port of llama.cpp's llm_tokenizer_spm) ───────────────

struct Spm {
    /// Mergeable pieces (normal + user-defined tokens) -> (id, score).
    pieces: HashMap<String, (u32, f32)>,
    byte_ids: [Option<u32>; 256],
    unk_id: Option<u32>,
    add_space_prefix: bool,
}

struct Bigram {
    score: f32,
    left: usize,
    size: usize,
}

impl PartialEq for Bigram {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Bigram {}
impl PartialOrd for Bigram {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Bigram {
    // Max-heap: higher score first, then leftmost.
    fn cmp(&self, o: &Self) -> Ordering {
        self.score
            .partial_cmp(&o.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| o.left.cmp(&self.left))
    }
}

struct Sym {
    start: usize,
    len: usize,
    prev: Option<usize>,
    next: Option<usize>,
}

impl Spm {
    fn new(
        tokens: &[String],
        types: &[i64],
        scores: &[f32],
        unk_id: Option<u32>,
        add_space_prefix: bool,
    ) -> Self {
        let mut pieces = HashMap::with_capacity(tokens.len());
        let mut byte_ids = [None; 256];
        for (i, t) in tokens.iter().enumerate() {
            match types[i] {
                TYPE_BYTE => {
                    if let Some(b) = parse_byte_token(t) {
                        byte_ids[b as usize] = Some(i as u32);
                    }
                }
                TYPE_NORMAL | TYPE_USER_DEFINED => {
                    pieces.insert(t.clone(), (i as u32, scores.get(i).copied().unwrap_or(0.0)));
                }
                _ => {}
            }
        }
        Self { pieces, byte_ids, unk_id, add_space_prefix }
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        let mut s = String::with_capacity(text.len() + 3);
        if self.add_space_prefix {
            s.push('\u{2581}');
        }
        s.push_str(&text.replace(' ', "\u{2581}"));

        let mut syms: Vec<Sym> = Vec::with_capacity(s.len());
        for (start, ch) in s.char_indices() {
            let i = syms.len();
            syms.push(Sym {
                start,
                len: ch.len_utf8(),
                prev: i.checked_sub(1),
                next: None,
            });
            if i > 0 {
                syms[i - 1].next = Some(i);
            }
        }

        let mut heap = BinaryHeap::new();
        let try_add = |heap: &mut BinaryHeap<Bigram>, syms: &[Sym], l: Option<usize>, r: Option<usize>| {
            let (Some(l), Some(r)) = (l, r) else { return };
            let piece = &s[syms[l].start..syms[r].start + syms[r].len];
            if let Some(&(_, score)) = self.pieces.get(piece) {
                heap.push(Bigram { score, left: l, size: piece.len() });
            }
        };
        for i in 1..syms.len() {
            try_add(&mut heap, &syms, Some(i - 1), Some(i));
        }

        while let Some(bg) = heap.pop() {
            let l = bg.left;
            let Some(r) = syms[l].next else { continue };
            // Stale entry: one side was merged since this bigram was queued.
            if syms[l].len == 0 || syms[r].len == 0 || syms[l].len + syms[r].len != bg.size {
                continue;
            }
            syms[l].len += syms[r].len;
            syms[r].len = 0;
            syms[l].next = syms[r].next;
            if let Some(n) = syms[r].next {
                syms[n].prev = Some(l);
            }
            try_add(&mut heap, &syms, syms[l].prev, Some(l));
            try_add(&mut heap, &syms, Some(l), syms[l].next);
        }

        let mut ids = Vec::with_capacity(syms.len());
        let mut cur = if syms.is_empty() { None } else { Some(0) };
        while let Some(i) = cur {
            let piece = &s[syms[i].start..syms[i].start + syms[i].len];
            match self.pieces.get(piece) {
                Some(&(id, _)) => ids.push(id),
                None => {
                    for &b in piece.as_bytes() {
                        if let Some(id) = self.byte_ids[b as usize].or(self.unk_id) {
                            ids.push(id);
                        }
                    }
                }
            }
            cur = syms[i].next;
        }
        ids
    }
}

/// Incremental UTF-8 decoder: holds back incomplete multi-byte sequences that are
/// split across tokens (common for emoji / CJK in byte-level vocabularies).
#[derive(Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    pub fn push(&mut self, bytes: &[u8], out: &mut String) {
        self.pending.extend_from_slice(bytes);
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    return;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[..valid]).unwrap());
                    match e.error_len() {
                        // Genuinely invalid bytes: replace and continue.
                        Some(bad) => {
                            out.push('\u{FFFD}');
                            self.pending.drain(..valid + bad);
                        }
                        // Incomplete sequence at the end: wait for more bytes.
                        None => {
                            self.pending.drain(..valid);
                            return;
                        }
                    }
                }
            }
        }
    }

    pub fn finish(&mut self, out: &mut String) {
        if !self.pending.is_empty() {
            out.push_str(&String::from_utf8_lossy(&self.pending));
            self.pending.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Prints token ids for each line of `SERAPH_TOK_TEXT` (a file) using the model at
    /// `SERAPH_TOK_MODEL`, for comparison against llama.cpp's `llama-tokenize`.
    /// Run: cargo test --release dump_tokens -- --ignored --nocapture
    #[test]
    #[ignore]
    fn dump_tokens() {
        let model = std::env::var("SERAPH_TOK_MODEL").expect("SERAPH_TOK_MODEL");
        let text = std::fs::read_to_string(std::env::var("SERAPH_TOK_TEXT").expect("SERAPH_TOK_TEXT")).unwrap();
        let mut f = std::fs::File::open(model).unwrap();
        let mut content = gguf_file::Content::read(&mut f).unwrap();
        let tok = Tok::from_gguf(&mut content).unwrap();
        for case in text.split("\n---\n") {
            let ids = tok.encode(case);
            println!("IDS {:?}", ids);
            let mut out = String::new();
            let mut s = Utf8Stream::default();
            for &id in &ids {
                s.push(tok.token_bytes(id), &mut out);
            }
            s.finish(&mut out);
            println!("ROUNDTRIP {}", out == case || out.trim_start() == case);
        }
    }
}
