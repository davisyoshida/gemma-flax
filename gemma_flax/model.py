from dataclasses import dataclass
from functools import partial

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

@dataclass
class GemmaConfig:
    n_vocab: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    num_layers: int
    ln_eps: float
    jax_dtype: str = 'float32'

    @property
    def dtype(self):
        return {
            'bfloat16': jnp.bfloat16,
            'float32': jnp.float32,
            'float16': jnp.float16,
        }[self.jax_dtype]

class GemmaModule(nn.Module):
    config: GemmaConfig

    def make_dense(self, size, name):
        return nn.Dense(size, name=name, use_bias=False, param_dtype=self.config.dtype)

class GemmaRMSNorm(GemmaModule):
    def setup(self):
        self.weight = self.param(
            'weight',
            partial(nn.initializers.zeros, dtype=self.config.dtype),
            (self.config.hidden_size,),
        )

    def __call__(self, x):
        orig_dtype = x.dtype
        x = x.astype(jnp.float32)
        x = x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.config.ln_eps)
        x = x.astype(orig_dtype)

        output = x * (self.weight + 1)
        return output

class GemmaMLP(GemmaModule):
    def setup(self):
        self.up_proj = self.make_dense(self.config.intermediate_size, name='up_proj')
        self.gate_proj = self.make_dense(self.config.intermediate_size, name='gate_proj')
        self.down_proj = self.make_dense(self.config.hidden_size, name='down_proj')

    def __call__(self, x):
        gate = jax.nn.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(up * gate)

def rotate_half(x):
    n = x.shape[-1] // 2
    x1 = x[..., :n]
    x2 = x[..., n:]
    return jnp.concatenate([-x2, x1], axis=-1)

def compute_freqs(head_dim, start_pos, n_pos, theta=10000, dtype=jnp.float32):
    positions = jnp.arange(n_pos) + start_pos
    inv_freq = 1 / (theta  ** (jnp.arange(0, head_dim, 2.0, dtype=jnp.float32) / head_dim))
    freqs = positions[:, None] * inv_freq[None, :]
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return (
        jnp.sin(emb).astype(dtype),
        jnp.cos(emb).astype(dtype)
    )

def apply_rotary_pos_emb(q_or_k, start_index):
    _, n_pos, head_dim = q_or_k.shape
    sin, cos = compute_freqs(head_dim, start_index, n_pos, dtype=q_or_k.dtype)
    emb = q_or_k * cos + rotate_half(q_or_k) * sin
    return emb

def _make_causal_mask(q_len, kv_len, n_past):
    mask = jnp.full((q_len, kv_len), -jnp.inf)
    mask = jnp.where(
        jnp.arange(q_len)[:, None] + n_past >= jnp.arange(kv_len)[None, :],
        0.,
        mask
    )

    return mask

class GemmaSelfAttention(GemmaModule):
    def setup(self):
        qkv_out = self.config.head_dim * (self.config.num_heads + 2 * self.config.num_kv_heads)
        self.qkv = self.make_dense(qkv_out, name='qkv_proj')
        self.proj = self.make_dense(self.config.hidden_size, name='o_proj')

    def __call__(self, x, kv_cache=None, kv_index=None):
        assert (kv_cache is None) == (kv_index is None)

        past_len = 0 if kv_index is None else kv_index

        q_len, _ = x.shape

        qkv = self.qkv(x)
        xq, xk, xv = jnp.split(qkv, [
            self.config.head_dim * self.config.num_heads,
            self.config.head_dim * (self.config.num_heads + self.config.num_kv_heads)
        ], axis=-1)

        xq = xq.reshape((q_len, self.config.num_heads, self.config.head_dim))
        xk = xk.reshape((q_len, self.config.num_kv_heads, self.config.head_dim))
        xv = xv.reshape((q_len, self.config.num_kv_heads, self.config.head_dim))

        xq = xq.transpose(1, 0, 2)
        xk = xk.transpose(1, 0, 2)
        xv = xv.transpose(1, 0, 2)

        xq = apply_rotary_pos_emb(xq, past_len)
        xk = apply_rotary_pos_emb(xk, past_len)


        if kv_cache is not None:
            kv_cache.shape == (2, xk.shape[0], kv_cache.shape[1], xk.shape[2])
            kv_cache = jax.lax.dynamic_update_slice(kv_cache, xk[None], (0, 0, kv_index, 0))
            kv_cache = jax.lax.dynamic_update_slice(kv_cache, xv[None], (1, 0, kv_index, 0))
            xk, xv = kv_cache

        kv_len = xk.shape[1]
        attention_mask = _make_causal_mask(q_len, kv_len, past_len)

        keys_per_query = self.config.num_heads // self.config.num_kv_heads
        split_queries = xq.reshape((keys_per_query, self.config.num_kv_heads, q_len, self.config.head_dim))
        logits = jnp.einsum('hjqd,hjkd->hjqk', split_queries, xk[None]) / jnp.sqrt(self.config.head_dim)
        logits += attention_mask
        attn = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(xv.dtype)

        attn_out = jnp.einsum('hvqk,hvkd->qhvd', attn, xv[None])
        attn_out = attn_out.reshape((q_len, (self.config.num_heads * self.config.head_dim)))
        h = self.proj(attn_out)
        return h, kv_cache

class GemmaBlock(GemmaModule):
    def setup(self):
        self.input_layernorm = GemmaRMSNorm(self.config)
        self.self_attn = GemmaSelfAttention(self.config)
        self.post_attention_layernorm = GemmaRMSNorm(self.config)
        self.mlp = GemmaMLP(self.config)

    def __call__(self, x, kv_cache, kv_index):
        residual = x
        h = self.input_layernorm(x)
        h, kv = self.self_attn(h, kv_cache, kv_index)

        h += residual
        residual = h

        h = self.post_attention_layernorm(h)
        h = self.mlp(h)

        return h + residual, kv

class GemmaModel(GemmaModule):
    def init_kv_cache(self, max_len):
        return jnp.zeros(
            (self.config.num_layers, 2, self.config.num_kv_heads, max_len, self.config.head_dim),
            dtype=self.config.dtype
        )

    def setup(self):
        self.tokens_emb = nn.Embed(self.config.n_vocab, self.config.hidden_size, param_dtype=self.config.dtype)
        self.trunk = nn.scan(
            nn.remat(GemmaBlock),
            variable_axes={'params': 0},
            split_rngs={'params': False},
            length=self.config.num_layers,
            in_axes=(0, flax.core.broadcast)
        )(self.config)
        self.norm = GemmaRMSNorm(self.config)

    def __call__(self, x, kv_cache=None, kv_index=None):
        h = self.tokens_emb(x)
        h *= self.config.hidden_size ** 0.5
        h_final, kvs = self.trunk(h, kv_cache, kv_index)

        pre_logits = self.norm(h_final)
        logits = self.tokens_emb.attend(pre_logits)
        ret = {'logits': logits}
        if kv_cache is not None:
            ret['kv_cache'] = kvs
            ret['kv_index'] = kv_index + x.shape[0]

        return ret
