//! Model implementations adapted from candle-transformers (MIT / Apache-2.0,
//! https://github.com/huggingface/candle, commit cfcbd71).
//!
//! Changes from upstream, all to make 14B models fit and stay fast on a 12 GB card:
//!
//! - The token-embedding table stays in system RAM ([`HostEmbedding`]) instead of
//!   being dequantized to f32 on the GPU. For a 14B Qwen model that table is
//!   ~3 GB (152k tokens x 5120 x 4 bytes). It is only a row lookup, so doing it
//!   on the CPU and copying the few rows needed costs almost nothing.
//! - The KV cache is stored in f16, halving its size. Each layer's cache is
//!   widened back to the compute dtype only while that layer runs.
//! - Grouped-query attention ([`gqa_attention`]) no longer copies every K/V head
//!   once per query head (`repeat_kv`). Upstream materialised the whole cache 5x
//!   per layer per token, which used ~100 MB of scratch per layer at 2.5k tokens
//!   and slowed generation down as the context grew.

pub mod qwen2;
pub mod qwen3;

use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Module, Result, Tensor};

/// dtype the KV cache is stored in.
pub const KV_DTYPE: DType = DType::F16;

/// Attention for `q` (b, heads, l, d) against `k`/`v` (b, kv_heads, kv_len, d),
/// where each group of `heads / kv_heads` consecutive query heads shares one K/V
/// head. Query heads are folded into the sequence axis so K/V are never copied.
/// `mask` is applied to scores shaped (b, heads, l, kv_len). Returns (b, heads, l, d).
pub fn gqa_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: impl FnOnce(Tensor) -> Result<Tensor>,
) -> Result<Tensor> {
    let (b, h, l, d) = q.dims4()?;
    let (_, kvh, kv_len, _) = k.dims4()?;
    let rep = h / kvh;
    let k = k.to_dtype(q.dtype())?.contiguous()?;
    let v = v.to_dtype(q.dtype())?.contiguous()?;
    let q = q.contiguous()?.reshape((b, kvh, rep * l, d))?;
    let scores = (q.matmul(&k.t()?)? * (1.0 / (d as f64).sqrt()))?;
    let scores = mask(scores.reshape((b, h, l, kv_len))?)?;
    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    let out = probs.reshape((b, kvh, rep * l, kv_len))?.matmul(&v)?;
    out.reshape((b, h, l, d))
}

#[derive(Debug, Clone)]
pub struct HostEmbedding {
    /// (vocab, hidden) f32 on the CPU.
    table: Tensor,
    hidden: usize,
    device: Device,
}

impl HostEmbedding {
    /// `weights` may live on any device; it is dequantized on the CPU.
    pub fn from_qtensor(weights: &QTensor, hidden: usize, device: &Device) -> Result<Self> {
        let table = weights.dequantize(&Device::Cpu)?;
        Ok(Self { table, hidden, device: device.clone() })
    }
}

impl Module for HostEmbedding {
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let dims = ids.dims().to_vec();
        let flat = ids.to_device(&Device::Cpu)?.flatten_all()?;
        let rows = self.table.index_select(&flat, 0)?;
        let mut shape = dims;
        shape.push(self.hidden);
        rows.reshape(shape)?.to_device(&self.device)
    }
}
