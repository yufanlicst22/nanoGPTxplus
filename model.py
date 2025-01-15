
import flax.linen as nn
from typing import Any, Optional, Tuple
import jax
from config.scalable import GPTConfig
import jax.numpy as jnp



class SelfAttention(nn.Module):

    num_heads: int
    dtype: Any = jnp.float32
    dropout_rate: float = 0.1
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, x, mask, deterministic=None):

        B, T, C = x.shape

        assert C % self.num_heads == 0
        head_dim = C // self.num_heads

        deterministic = nn.merge_param('deterministic', self.deterministic, deterministic)

        # Construct qkv together a one single tensor
        qkv=nn.DenseGeneral((3, self.num_heads, head_dim), dtype=self.dtype)(x) # out: (B, T, 3, num_heads, head_dim)

        # split qkv at dim=2 into q, k, v of shape (B, T, num_heads, head_dim)
        q, k, v = [x.squeeze(axis=2) for x in jnp.array_split(qkv, 3, axis=2)] # out: (B, T, num_heads, head_dim)

        # normalizaing scalar in the self-attention formula
        scale = 1.0 / jnp.sqrt(head_dim).astype(self.dtype)

        # q, k: (B, T, num_heads, head_dim)
        # transpose q, v so that batch dimensions are in front: (B, num_heads, T, head_dim)@(B, num_heads, head_dim, T)
        attn = q.transpose((0,2,1,3))@k.transpose((0,2,3,1)) * scale # out: (B, num_heads, T, T)
        attn = jnp.where(mask, attn, jnp.finfo(self.dtype).min) # out: (B, num_heads, T, T)

        # input attn: (B, num_heads, T, T); apply softmax at last dimension (default)
        attn = jax.nn.softmax(attn).astype(self.dtype) # out: (B, num_heads, T, T)
        attn = nn.Dropout(self.dropout_rate)(attn, deterministic=deterministic) # out: (B, num_heads, T, T)

        # v: (B, T, num_heads, head_dim), attn: (B, num_heads, T, T)
        x=(attn@(v.transpose((0,2,1,3)))).transpose(0,2,1,3).reshape(B,T,C) # out: (B,T,C)
        x = nn.Dense(C, dtype=self.dtype, name='c_proj')(x) # out: (B,T,C)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic) # out: (B,T,C)

        return x

class MLP(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x, deterministic=None):
        B, T, C = x.shape
        x = nn.Dense(4 * C, dtype=self.config.dtype, name='c_fc')(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dense(C, dtype=self.config.dtype, name='c_proj')(x)
        x = nn.Dropout(self.config.dropout_rate)(x, deterministic)
        return x

class Block(nn.Module):
    config: GPTConfig

    def setup(self):
        self.ln_1 = nn.LayerNorm(epsilon=1e-5, dtype=self.config.dtype)
        self.attn = SelfAttention(self.config.num_heads,
                                  self.config.dtype,
                                  dropout_rate=self.config.dropout_rate)
        self.ln_2 = nn.LayerNorm(epsilon=1e-5, dtype=self.config.dtype)
        self.mlp = MLP(self.config)

    def __call__(self, x, mask=None, deterministic=None):
        x = x + self.attn(self.ln_1(x), mask, deterministic)
        x = x + self.mlp(self.ln_2(x), deterministic)
        return x

class GPT(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, idx, deterministic=None):

        B, T = idx.shape
        wte = nn.Embed(self.config.vocab_size, self.config.num_embeds, dtype=self.config.dtype, name='wte')
        wpe = nn.Embed(self.config.block_size, self.config.num_embeds, dtype=self.config.dtype, name='wpe')
        token_embed = wte(idx)      # out: (B, T, num_embeds)
        pos_embed = wpe(jnp.arange(0, T)[None])        # out: (1, T, num_embeds)
        x = nn.Dropout(self.config.dropout_rate)(token_embed + pos_embed, deterministic) # out: (B,T,num_embeds)

        attn_mask = nn.make_causal_mask(idx, dtype=bool)
        for i in range(self.config.num_layers):
            x = Block(self.config, name=str(i))(x, attn_mask, deterministic=deterministic) # out: (B,T,num_embeds)

        x = nn.LayerNorm(1e-5, dtype=self.config.dtype, name='ln_f')(x) # out: (B,T,num_embeds)
        logits = wte.attend(x)
        return logits

    def init(self, rng):
        tokens = jnp.zeros((self.config.batch_size, self.config.block_size), dtype=jnp.uint16)
        params = jax.jit(super().init, static_argnums=(2,))(rng, tokens, True)
        return params