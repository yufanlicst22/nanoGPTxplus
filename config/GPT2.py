
from typing import Any, Optional, Tuple
from jax.typing import DTypeLike
from dataclasses import dataclass
import jax.numpy as jnp

@dataclass(frozen=True)
class GPTConfig:

    # How many tokens per param
    CHINCHILLA_MULTIPLIER = 20

    # Model hyperparameters
    num_layers: int = 12 #12
    num_heads: int = 12 #12
    num_embeds: int = 768
    vocab_size=50257
    dtype_1: DTypeLike = jnp.float32
    dtype_2: DTypeLike = jnp.bfloat16
    dropout_rate: float = 0.1
    block_size: int = 1024 # context length

    # Batch size
    batch_size: int = 128
    grad_accum_steps: int = 4 # microbatch size = batch_size / (num_device * grad_accum_steps)

    # Calculate training durations
    token_per_batch = block_size * batch_size
    n_params =  (12*num_embeds**2 + 13*num_embeds) * (num_layers) + vocab_size * num_embeds + 2*num_embeds
    num_steps = int(CHINCHILLA_MULTIPLIER * n_params // token_per_batch)

    # Optimizer config
    independent_weight_decay = 8 / num_steps
    warmup_steps = int(0.1*num_steps)

    # Eval
    eval_batch_size = 8
    eval_every_steps = 200
    eval_steps = 100

    # Check-pointing
    save_checkpoint: bool = False
    checkpoint_every_steps = 200

    # Peak learning rate
    peak_lr = 1e-3

    # Remat
    remat_attn: bool = True

    # Tensorboard profiling
    enable_tbprof: bool = False
    profile_at_step: int = 5 # start profiling at which step
    profile_num_steps: int = 3 # profile for how many steps
    tb_logdir: str = "./tb_logs/run3" # store output dir
    
