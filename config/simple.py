
from typing import Any, Optional, Tuple
from dataclasses import dataclass

@dataclass(frozen=True)
class GPTConfig:

    # How many tokens per param
    CHINCHILLA_MULTIPLIER = 20

    # Model hyperparameters
    num_layers: int = 1
    num_heads: int = 2
    num_embeds: int = 16
    vocab_size=32000
    dtype: Optional[str] = None
    dropout_rate: float = 0.1
    block_size: int = 2 # context length

    # Batch size
    batch_size=8

    # Calculate training durations
    token_per_batch = block_size * batch_size
    n_params =  (num_embeds ** 2) * (12 * num_layers) + vocab_size * num_embeds
    num_steps = int(CHINCHILLA_MULTIPLIER * n_params // token_per_batch)

    # Optimizer config
    independent_weight_decay = 8 / num_steps
    warmup_steps = int(0.1*num_steps)

    # Eval
    eval_batch_size = 2
    eval_every_steps = 400
    eval_steps = 100

    # Check-pointing
    checkpoint_every_steps = 200

    # Peak learning rate
    peak_lr = 1e-3
