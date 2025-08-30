from typing import Tuple, Any
from functools import partial
import os
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from flax.training.train_state import TrainState
import optax
import flax.serialization
import multiprocessing
import pickle
import time
from config.GPT2 import GPTConfig
from model import GPT
#from data.datanoam import NoamPackedIterableDataset

#from transformers import BertTokenizer 
from torch.utils.data import DataLoader, IterableDataset
#from datasets import load_dataset
import torch
from torch.multiprocessing import freeze_support

from data.fineweb_streaming import get_dataloader as get_fw_dataloader

# === Training ===
@partial(jax.pmap, axis_name="batch")
def train_step(state: TrainState, key, tokens) -> Tuple[jnp.ndarray, TrainState]:
    dropout_key = jax.random.fold_in(key, state.step)

    def loss_fn(params: FrozenDict) -> jnp.ndarray:
        logits = state.apply_fn(params, tokens[:, :-1], False, rngs={"dropout": dropout_key})
        loss = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), tokens[:, 1:]).mean()
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")  # Average gradients across devices
    loss = jax.lax.pmean(loss, axis_name="batch")  # Average loss across devices
    new_state = state.apply_gradients(grads=grads)
    return loss, new_state

def init_train_state(key, config) -> TrainState:
    gpt = GPT(config)
    params = gpt.init(key)

    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.peak_lr,
        warmup_steps=config.warmup_steps,
        decay_steps=config.num_steps,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate, 0.9, 0.95, weight_decay=0.01),
        optax.apply_every(1),
    )
    train_state = TrainState.create(
        apply_fn=gpt.apply,
        params=params,
        tx=optimizer,
    )
    return train_state

# === Evaluation ===
@partial(jax.pmap, axis_name="batch")
def eval_step(state: TrainState, tokens) -> jnp.ndarray:
    logits = state.apply_fn(state.params, tokens[:, :-1], True)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), tokens[:, 1:])
    loss = jax.lax.pmean(loss, axis_name="batch")
    return loss

def evaluate(state: TrainState, steps: int, eval_iterator: Any) -> jnp.ndarray:
    losses = []
    for step in range(steps):
        tokens = jnp.array(next(eval_iterator))
        tokens_sharded = shard_data(tokens)
        loss = eval_step(state, tokens_sharded)
        losses.append(loss)
    return jnp.mean(jnp.stack(losses))

# === Sharding ===
def shard_data(data):
    n_devices = jax.local_device_count()
    return data.reshape(n_devices, -1, *data.shape[1:])

# === Manage check points ===
def save_checkpoint(train_state, train_iterator, step, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.msgpack")
    iterator_state_path = os.path.join(checkpoint_dir, f"train_iterator_state_{step}.pkl")

    # Save train state
    with open(checkpoint_path, "wb") as f:
        f.write(flax.serialization.to_bytes(train_state))

    # Save train iterator state
    with open(iterator_state_path, "wb") as f:
        pickle.dump(train_iterator.get_state(), f)

def load_checkpoint(step, checkpoint_dir="checkpoints"):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.msgpack")
    iterator_state_path = os.path.join(checkpoint_dir, f"train_iterator_state_{step}.pkl")

    # Load train state
    with open(checkpoint_path, "rb") as f:
        train_state_bytes = f.read()
    # Deserialize train_state using an initialized dummy TrainState
    train_state = flax.serialization.from_bytes(init_train_state(jax.random.PRNGKey(0), config), train_state_bytes)

    # Load train iterator state
    with open(iterator_state_path, "rb") as f:
        iterator_state = pickle.load(f)

    return train_state, iterator_state

# ==== Profiling ====
def print_device_memory(prefix=""):

    # Make sure all pending work finished so numbers are meaningful
    jax.block_until_ready(jnp.array(0))

    def fmt(b):
        return f"{b/(1024**3):.2f} GiB"

    for i, dev in enumerate(jax.local_devices()):
        try:
            stats = dev.memory_stats()  # dict; keys vary by backend
            used  = (stats.get("bytes_in_use")
                     or stats.get("memory_in_use")
                     or stats.get("current_bytes")
                     or 0)
            peak  = (stats.get("peak_bytes_in_use")
                     or stats.get("peak_memory_in_use")
                     or stats.get("peak_bytes")
                     or None)
            limit = (stats.get("bytes_limit")
                     or stats.get("bytes_reserved")
                     or stats.get("total_bytes")
                     or None)
            line = f"[{prefix}] dev{i} {dev.device_kind}: used={fmt(used)}"
            if peak:  line += f", peak={fmt(peak)}"
            if limit: line += f", limit={fmt(limit)}"
            print(line)
        except Exception as e:
            print(f"[{prefix}] dev{i} {dev.device_kind}: memory_stats() not available ({e})")


def main():
    # ===== Initialization =====
    config = GPTConfig
    key = jax.random.PRNGKey(0)
    key, subkey_init_train_state = jax.random.split(key)
    train_state = init_train_state(subkey_init_train_state, config)

    # Checkpointing
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Variables
    timestamp=[]
    val_loss_arr=[]
    loss = 0.00

    # Profiling
    elapsed_time = data_time = train_time = eval_time = check_time = 0.0
    def log_memory():
        for key in jax.local_devices()[0].memory_stats():
            print(key, f"{jax.local_devices()[0].memory_stats()[key]/(1024**3):.2f}GB")

    def get_train_state_memory_size(train_state):
        # Traverse the tree and compute the size of each parameter
        param_sizes = jax.tree_util.tree_map(lambda x: x.size * x.itemsize if isinstance(x, jnp.ndarray) else 0, train_state)
        # Sum all parameter sizes
        total_size_bytes = sum(jax.tree_util.tree_flatten(param_sizes)[0])
        return total_size_bytes/(1024**3)

    # Replicate train state for multi-device training
    train_state = jax.device_put_replicated(train_state, jax.local_devices())

    # Create data iterator
    def get_iter(config, split):

        is_train = (split == "train")

        dataloader = get_fw_dataloader(
            batch_size=config.batch_size,
            num_workers=4,                     # safest with JAX
            repo="HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            block_size=config.block_size,      # must match your model
            shuffle_buffer=10_000 if is_train else 5_000,
            max_tokens=None,
            seed=1337 if is_train else 4242,
            take_docs=None,
        )
        di = iter(dataloader)
        return di
    train_iterator = get_iter(config, 'train')
    val_iterator = get_iter(config, 'test')


    print('==== Train Specs ====')
    print(f"CPU cores: {multiprocessing.cpu_count()}, local devices: {jax.local_device_count()}")
    print('N:', f'{config.n_params // 10**6:.2f} M')
    print('Width:', f'{config.num_embeds}')
    print('Depth:', f'{config.num_layers}')
    print('Batch size:', f'{config.batch_size}')
    print('Context length:', f'{config.block_size}')
    print('Number of heads:', f'{config.num_heads}')
    print('Train steps: ', f'{config.num_steps / 10**3:.3f} K')
    print('Tokens per step: ', f'{config.token_per_batch // 10**3} K')
    print('=======================')
    print(' ')
    print('==== Memory Estimates (per device) ====')
    # bytes per dtype
    _b_param = int(jnp.dtype(config.dtype_1).itemsize)   # usually fp32 -> 4
    _b_act   = int(jnp.dtype(config.dtype_2).itemsize)   # usually bf16 -> 2

    # persistent (per replica)
    _param_bytes = int(config.n_params) * _b_param                           # parameters stored in dtype_1
    _grad_bytes  = int(config.n_params) * int(jnp.dtype(jnp.float32).itemsize)  # grads kept in fp32
    _opt_bytes   = 2 * int(config.n_params) * int(jnp.dtype(jnp.float32).itemsize)  # Adam m+v (fp32)

    # activations (mixed precision, no recompute):  m_act = L * seq * bs_per_device * h * (34 + 5*heads*seq/h)
    _replicas = max(1, jax.local_device_count())
    _bs   = int(config.batch_size // _replicas)  # per-device microbatch
    _seq  = int(config.block_size)
    _h    = int(config.num_embeds)
    _nh   = int(config.num_heads)
    _act_bytes = (config.num_layers * _seq * _bs * _h * (34.0 + (5.0 * _nh * _seq) / _h)) * _b_act

    print('Params:',      f"{config.n_params/1e6:.2f} M, {_param_bytes/(1024**3):.2f} GiB @ {jnp.dtype(config.dtype_1).name}")
    print('Grad:',        f"{_grad_bytes/(1024**3):.2f} GiB (fp32)")
    print('Opt (Adam):',  f"{_opt_bytes/(1024**3):.2f} GiB (m+v fp32)")
    print('Activations:', f"{_act_bytes/(1024**3):.2f} GiB (dtype={jnp.dtype(config.dtype_2).name}, no recompute)")
    print('=======================')



    # ===== Train loop =====
    start_time = time.time()
    step, use_checkpoint = 0, False
    is_save_checkpoint = False
    profiling = True
    while step <= config.num_steps:

        # Load check points
        if use_checkpoint:
            train_state, train_iterator_state = load_checkpoint(step, checkpoint_dir)
            train_iterator.set_state(train_iterator_state)
            checkpoint = False

        # Evaluation
        if step % config.eval_every_steps == 0:

            start_tmp = time.time()
            val_loss = evaluate(train_state, config.eval_steps, val_iterator)
            if profiling:
                val_loss.block_until_ready()
                eval_time += time.time() - start_tmp

            val_loss_arr.append(val_loss)
            elapsed_time = time.time() - start_time

            train_loss = loss[0] if (jax.local_device_count()>1 and step>0) else loss

            print(f"step {step}, time: {elapsed_time:.1f}s, train_loss: {train_loss}, eval_loss: {val_loss:.4f}")
            if profiling:
                print(f"train:{train_time/elapsed_time*100:.1f}%, eval:{eval_time/elapsed_time*100:.1f}%, data:{data_time/elapsed_time*100:.1f}%, chkpoint:{check_time/elapsed_time*100:.1f}%")
                print_device_memory(prefix=f"step {step}")

        # Generate new RNG keys for all devices
        keys = jax.random.split(key, jax.local_device_count())

        # Prepare and shard the data
        start_tmp = time.time()
        tokens = jnp.array(next(train_iterator))
        tokens_sharded = shard_data(tokens)  # Shard tokens for pmap
        if profiling:
            tokens_sharded.block_until_ready()
            data_time += time.time() - start_tmp

        # Train step
        start_tmp = time.time()
        loss, train_state = train_step(train_state, keys, tokens_sharded)
        step += 1
        if profiling:
            loss.block_until_ready()
            train_time += time.time() - start_tmp

        # Save checkpoint
        start_tmp = time.time()
        if is_save_checkpoint and step % config.checkpoint_every_steps == 0:
            save_checkpoint(train_state, train_iterator, step, checkpoint_dir)
        if profiling:
            check_time += time.time() - start_tmp

if __name__ == "__main__":
    freeze_support()
    main()