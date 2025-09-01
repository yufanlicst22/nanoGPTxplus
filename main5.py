from typing import Tuple, Any
from functools import partial
import glob, os
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
import flax.linen as nn
from torch.utils.data import DataLoader, IterableDataset
#from datasets import load_dataset
import torch
from torch.multiprocessing import freeze_support
from flax import jax_utils

from data.fineweb2 import get_dataloader as get_fw_dataloader

from datetime import datetime
from jax.profiler import StepTraceAnnotation
import jax.profiler as jprof

from typing import Optional

# === Training ===
@partial(jax.pmap, axis_name="batch")
def train_step(state: TrainState, key, tokens) -> Tuple[jnp.ndarray, TrainState]:
    n_micro = tokens.shape[0]

    base_key = jax.random.fold_in(key, state.step)
    micro_keys = jax.random.split(base_key, n_micro)

    grads_zero = jax.tree_util.tree_map(jnp.zeros_like, state.params)
    def micro_body(carry, inputs):
        grads_accum, loss_accum = carry
        tok_i, k_i = inputs  # [micro_bsz_per_device, seq_len], PRNGKey

        def loss_fn(params: FrozenDict) -> jnp.ndarray:
            logits = state.apply_fn(params, tok_i[:, :-1], False, rngs={"dropout": k_i})
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits.astype(jnp.float32), tok_i[:, 1:]
            ).mean()  # mean over (micro_bsz * (seq-1))
            return loss
        
        loss_i, grads_i = jax.value_and_grad(loss_fn)(state.params)
        grads_accum = jax.tree_util.tree_map(lambda a, b: a + b, grads_accum, grads_i)
        loss_accum = loss_accum + loss_i
        return (grads_accum, loss_accum), None

    (grads_sum, loss_sum), _ = jax.lax.scan(
        micro_body,
        (grads_zero, jnp.array(0.0, dtype=jnp.float32)),
        (tokens, micro_keys),
    )

    grads = jax.tree_util.tree_map(lambda g: g / n_micro, grads_sum)
    loss = loss_sum / n_micro

    grads = jax.lax.pmean(grads, axis_name="batch")
    loss  = jax.lax.pmean(loss,  axis_name="batch")
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

def shard_microbatches(data, grad_accum_steps: int):

    n_devices = jax.local_device_count()
    b = data.shape[0]
    assert b % (n_devices * grad_accum_steps) == 0, (
        f"Global batch {b} must be divisible by n_devices({n_devices}) * "
        f"grad_accum_steps({grad_accum_steps})."
    )
    per_dev_micro = b // (n_devices * grad_accum_steps)
    return data.reshape(n_devices, grad_accum_steps, per_dev_micro, *data.shape[1:])


# === Manage check points ===
def save_checkpoint(train_state, train_loader, step, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.msgpack")
    loader_state_path = os.path.join(checkpoint_dir, f"dataloader_state_{step}.pkl")

    # Save TrainState (host copy)
    host_state = jax_utils.unreplicate(train_state) if isinstance(train_state, TrainState) else train_state
    with open(checkpoint_path, "wb") as f:
        f.write(flax.serialization.to_bytes(host_state))

    # Save DataLoader state if supported (StatefulDataLoader)
    dl_state = train_loader.state_dict() if hasattr(train_loader, "state_dict") else {}
    with open(loader_state_path, "wb") as f:
        pickle.dump(dl_state, f)

def load_checkpoint(step, config, checkpoint_dir="checkpoints"):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.msgpack")
    loader_state_path = os.path.join(checkpoint_dir, f"dataloader_state_{step}.pkl")

    template = init_train_state(jax.random.PRNGKey(0), config)
    with open(checkpoint_path, "rb") as f:
        train_state_bytes = f.read()
    host_state = flax.serialization.from_bytes(template, train_state_bytes)
    train_state = jax_utils.replicate(host_state)

    with open(loader_state_path, "rb") as f:
        loader_state = pickle.load(f)

    return train_state, loader_state

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


def _block_tree(x):
    # Ensures all async work is finished so the trace is complete/ordered.
    return jax.tree.map(lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a, x)

def _latest_tb_profile_host_dir(tb_logdir: str) -> Optional[str]:
    base = os.path.join(tb_logdir, "plugins", "profile")
    run_dirs = [d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)]
    if not run_dirs:
        return None
    # Choose the newest run by modification time (safer than lexicographic sort)
    latest_run = max(run_dirs, key=os.path.getmtime)
    host_dirs = [d for d in glob.glob(os.path.join(latest_run, "*")) if os.path.isdir(d)]
    if not host_dirs:
        # Some setups write directly into the timestamp folder; fall back to it
        return latest_run
    # Prefer the host dir that already contains trace.json.gz or xplane.pb
    for d in host_dirs:
        files = set(os.listdir(d))
        if "trace.json.gz" in files or "xplane.pb" in files:
            return d
    # Otherwise just pick the first host dir
    return host_dirs[0]


# ==== Main Train Loop ====

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
    
    # ===== TensorBoard profiling settings =====
    tb_logdir = config.tb_logdir
    profile_at_step  = config.profile_at_step
    profile_num_steps = config.profile_num_steps
    tracing = False


    # Replicate train state for multi-device training
    train_state = jax.device_put_replicated(train_state, jax.local_devices())

    # Create data iterator
    def get_loader(config, split):
        is_train = (split == "train")
        return get_fw_dataloader(
            batch_size=config.batch_size,
            num_workers=4,                      
            stateful=True,                      
            repo="HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            block_size=config.block_size,
            shuffle_buffer=10_000 if is_train else 5_000,
            max_tokens=None,
            seed=1337 if is_train else 4242,
            take_docs=None,
        )

    train_loader = get_loader(config, 'train')
    val_loader   = get_loader(config, 'test')
    train_iterator = iter(train_loader)
    val_iterator   = iter(val_loader)


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
    _bs = int(config.batch_size // (_replicas * config.grad_accum_steps))  # per-device *micro*batch
    _seq  = int(config.block_size)
    _h    = int(config.num_embeds)
    _nh   = int(config.num_heads)
    _act_bytes_attn = (config.num_layers * _seq * _bs * _h * ( (5.0 * _nh * _seq) / _h)) * _b_act
    _act_bytes_others = (config.num_layers * _seq * _bs * _h * 34.0) * _b_act

    print('Params:',      f"{_param_bytes/(1024**3):.2f} GiB @ {jnp.dtype(config.dtype_1).name}")
    print('Grad:',        f"{_grad_bytes/(1024**3):.2f} GiB (fp32)")
    print('Opt (Adam):',  f"{_opt_bytes/(1024**3):.2f} GiB (m+v fp32)")
    print('Activations (attn square tensor):', f"{_act_bytes_attn/(1024**3):.2f} GiB (dtype={jnp.dtype(config.dtype_2).name}, no recompute)")
    print('Activations (others):', f"{_act_bytes_others/(1024**3):.2f} GiB (dtype={jnp.dtype(config.dtype_2).name}, no recompute)")
    print('=======================')


    # ===== Train loop =====
    start_time = time.time()
    step, use_checkpoint = 0, False
    profiling = True
    while step <= config.num_steps:
        print(step)

        # TensorBoard profiling: set prfile_at_step to be negative to avoid tracing
        if (config.enable_tbprof) and (not tracing) and (step == profile_at_step):
            os.makedirs(tb_logdir, exist_ok=True)
            print(f"[Profiler] Starting trace at step {step} -> {tb_logdir}")
            jprof.start_trace(tb_logdir)
            tracing = True

        # Load check points
        if use_checkpoint:
            train_state, loader_state = load_checkpoint(step, config, checkpoint_dir)
            train_loader.load_state_dict(loader_state)
            train_iterator = iter(train_loader)
            use_checkpoint = False

        # Evaluation
        if step % config.eval_every_steps == 0:
            with StepTraceAnnotation("eval", step_num=step):
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
        key, step_key = jax.random.split(key)
        keys = jax.random.split(step_key, jax.local_device_count())

        # Prepare and shard the data
        with StepTraceAnnotation("data", step_num=step):
            start_tmp = time.time()
            tokens = jnp.array(next(train_iterator))
            tokens_sharded = shard_microbatches(tokens, config.grad_accum_steps)
            if profiling:
                tokens_sharded.block_until_ready()
                data_time += time.time() - start_tmp

        # Train step
        with StepTraceAnnotation("train", step_num=step):
            start_tmp = time.time()
            loss, train_state = train_step(train_state, keys, tokens_sharded)
            step += 1
            if profiling:
                loss.block_until_ready()
                train_time += time.time() - start_tmp

        # Save checkpoint
        with StepTraceAnnotation("checkpoint", step_num=step):
            start_tmp = time.time()
            if config.save_checkpoint and step % config.checkpoint_every_steps == 0:
                save_checkpoint(train_state, train_loader, step, checkpoint_dir)
            if profiling:
                check_time += time.time() - start_tmp

        # ---- stop trace after N profiled steps ------------------------------
        if (config.enable_tbprof) and (tracing) and (step >= profile_at_step + profile_num_steps):
            # ensure all device work is done before closing the trace
            _block_tree((loss, train_state))
            jprof.stop_trace()
            tracing = False

            host_dir = _latest_tb_profile_host_dir(tb_logdir)
            if host_dir is None: host_dir = tb_logdir
            jprof.save_device_memory_profile(os.path.join(host_dir, "device_memory_profile.pb"))
            
            print(f"[Profiler] Trace finished at step {step}; open TensorBoard with:\n"
                  f"  tensorboard --logdir {tb_logdir}\n"
                  f"Then visit the 'Profile' tab → Trace Viewer / Memory.")

if __name__ == "__main__":
    freeze_support()
    main()