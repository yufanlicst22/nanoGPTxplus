
from typing import Any, Optional, Tuple
from dataclasses import dataclass

@dataclass(frozen=True)
class GPTConfig:

    # Scaling ladder
    #   model size = (num_embeds ** 2) * (12 * num_layers) + vocab_size * num_embeds = 244*1024**2 * k ** 3 + vocab_size * 1024 * k
    #   The ladder scales number of heads, depth, and width linearly with k
    k=0.25

    # How many tokens per param
    CHINCHILLA_MULTIPLIER = 20

    # Model hyperparameters
    num_layers: int = int(12*k) # number of transformer blocks
    num_heads: int = int(16*k) # multi-heads in one block
    num_embeds: int = 64*num_heads # embedding size
    vocab_size=32000
    dtype: Optional[str] = None
    dropout_rate: float = 0.1
    block_size: int = 128 # context length

    # Batch size
    #batch_size=int(64*k**2)
    batch_size=512

    # Calculate training durations
    token_per_batch = block_size * batch_size
    n_params =  (num_embeds ** 2) * (12 * num_layers) + vocab_size * num_embeds
    num_steps = int(CHINCHILLA_MULTIPLIER * n_params // token_per_batch)

    # Optimizer config
    independent_weight_decay = 8 / num_steps
    warmup_steps = int(0.1*num_steps)

    # Eval
    eval_batch_size = int(64*k**2)
    eval_every_steps = 400
    eval_steps = 100

    # Check-pointing
    checkpoint_every_steps = 200

    # Peak learning rate
    BASE_WIDTH = 1024
    BASE_DEPTH = 12
    BASE_CONTEXT_LENGTH = 2048
    BASE_BATCH_SIZE = 64
    BASE_TOKENS_PER_BATCH = BASE_CONTEXT_LENGTH * BASE_BATCH_SIZE
    BASE_OVERTRAINING = 1.
    BASE_LR = 2/1024

    WIDTH_EXPONENT = -1.
    NUM_LAYER_EXPONENT = -0.5
    TOKENS_PER_BATCH_EXPONENT = 0.5
    OVERTRAINING_EXPONENT = -0.25
    JOINT_FIT_EXPONENT = -1.

    # The below approach basically comes from:
    # (i)     Fixing a base model config (i.e. BASE_WIDTH,...) and do grid search on lr to get BASE_LR; 
    # (ii)    Assuming the optimal lr follows the following form and find the exponent 
    #         by freezing all but one hyperparameter (say num_embeds) at base config 
    # (iii)   Try different vals of that hyperparameter (num_embeds) and find optimal lr for each
    # (iv)    Fit the law
    peak_lr = BASE_LR * (num_embeds / BASE_WIDTH) ** WIDTH_EXPONENT \
      * (num_layers / BASE_DEPTH) ** NUM_LAYER_EXPONENT \
      * (token_per_batch / BASE_TOKENS_PER_BATCH) ** TOKENS_PER_BATCH_EXPONENT \
      * (CHINCHILLA_MULTIPLIER/20 / BASE_OVERTRAINING) ** OVERTRAINING_EXPONENT

    # Alterantively, if the scaling is done at chinchilla optimal, a simpler formula can be used (but it is less robust if we want to account for overtraining etc)
    # This one is easy, we have these scales k and for each we find optimal lr and fit below
    # peak_lr = BASE_LR * k ** JOINT_FIT_EXPONENT
