
import flax.linen as nn
from typing import Any, Optional, Tuple
import jax
from config.scalable import GPTConfig
import jax.numpy as jnp



class CausalSelfAttention(nn.Module):
    config: GPTConfig

    def setup(self):
        assert self.config.num_embeds % self.config.num_heads == 0
        self.c_attn = nn.Dense(3 * self.config.num_embeds, dtype=self.config.dtype_2, param_dtype=self.config.dtype_1, name='c_attn')
        self.c_proj = nn.Dense(self.config.num_embeds, dtype=self.config.dtype_2, param_dtype=self.config.dtype_1, name='c_proj')
        self.c_do_1 = nn.Dropout(self.config.dropout_rate)
        self.c_do_2 = nn.Dropout(self.config.dropout_rate)

    def __call__(self, x, mask, deterministic=None):
        B, T, C = x.shape 
        
        qkv = self.c_attn(x)
        q, k, v = jnp.split(qkv, 3, axis=2)
        k = k.reshape(B, T, self.config.num_heads, C // self.config.num_heads).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        q = q.reshape(B, T, self.config.num_heads, C // self.config.num_heads).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        v = v.reshape(B, T, self.config.num_heads, C // self.config.num_heads).transpose(0, 2, 1, 3)  # (B, nh, T, hs)

        scale = jnp.array(1.0 / jnp.sqrt(k.shape[-1]), dtype=self.config.dtype_2)
        att = (q @ k.transpose(0, 1, 3, 2)) * scale # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = att.astype(self.config.dtype_1)
        att = jnp.where(mask, att, jnp.finfo(self.config.dtype_1).min)
        att = jax.nn.softmax(att, axis=-1).astype(self.config.dtype_2)
        att = self.c_do_1(att, deterministic)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)  # re-assemble all head outputs side by side

        y = self.c_proj(y)
        y = self.c_do_2(y, deterministic)
        return y

class MLP(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x, deterministic=None):
        B, T, C = x.shape
        x = nn.Dense(4 * C, dtype=self.config.dtype_2, param_dtype=self.config.dtype_1, name='c_fc')(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dense(C, dtype=self.config.dtype_2, param_dtype=self.config.dtype_1, name='c_proj')(x)
        x = nn.Dropout(self.config.dropout_rate)(x, deterministic)
        return x

class Block(nn.Module):
    config: GPTConfig

    def setup(self):
        self.ln_1 = nn.LayerNorm(epsilon=1e-5, dtype=self.config.dtype_1)
        if getattr(self.config, "remat_attn", True): 
            self.attn = nn.remat(CausalSelfAttention, static_argnums=(3,), prevent_cse=True)(self.config)
        else:
            self.attn = CausalSelfAttention(self.config)
        self.ln_2 = nn.LayerNorm(epsilon=1e-5, dtype=self.config.dtype_1)
        self.mlp = MLP(self.config)

    def __call__(self, x, mask=None, deterministic=None):
        x = x + self.attn(self.ln_1(x).astype(self.config.dtype_2), mask, deterministic)
        x = x + self.mlp(self.ln_2(x).astype(self.config.dtype_2), deterministic)
        return x

class GPT(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, idx, deterministic=None):

        B, T = idx.shape
        wte = nn.Embed(self.config.vocab_size, self.config.num_embeds, dtype=self.config.dtype_2, param_dtype=self.config.dtype_1, name='wte')
        wpe = nn.Embed(self.config.block_size, self.config.num_embeds, dtype=self.config.dtype_2, param_dtype=self.config.dtype_1, name='wpe')
        token_embed = wte(idx)      # out: (B, T, num_embeds)
        pos_embed = wpe(jnp.arange(0, T)[None])        # out: (1, T, num_embeds)
        x = nn.Dropout(self.config.dropout_rate)(token_embed + pos_embed, deterministic) # out: (B,T,num_embeds)

        attn_mask = nn.make_causal_mask(idx, dtype=bool)
        for i in range(self.config.num_layers):
            x = Block(self.config, name=str(i))(x, attn_mask, deterministic=deterministic) # out: (B,T,num_embeds)

        x = nn.LayerNorm(1e-5, dtype=self.config.dtype_1, name='ln_f')(x) # out: (B,T,num_embeds)
        logits = nn.Dense(self.config.vocab_size, use_bias=False, dtype=self.config.dtype_2, param_dtype=self.config.dtype_1, name='lm_head')(x)
        return logits

    def init(self, rng):
        tokens = jnp.zeros((self.config.batch_size, self.config.block_size), dtype=jnp.uint16)
        params = jax.jit(super().init, static_argnums=(2,))(rng, tokens, True)
        return params