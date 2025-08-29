# data/fineweb_streaming.py
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import torch, tiktoken, random, itertools, collections

class FineWebPacked(IterableDataset):
    """
    Streams text -> token ids -> packed contiguous blocks of size `block_size`.
    Yields (x, y) where y is x shifted by 1.
    """
    def __init__(
        self,
        name="sample-10BT",           # or a specific dump like "CC-MAIN-2024-10"
        repo="HuggingFaceFW/fineweb-edu",
        block_size=1024,
        eos=True,
        shuffle_buffer=10_000,
        seed=1337,
        max_tokens=None,              # e.g., 50_000_000 for ~50M tokens quick run
        take_docs=None                # e.g., 200_000 docs quick dry-run
    ):
        ds = load_dataset(repo, name=name, split="train", streaming=True)
        if shuffle_buffer:
            ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)  # HF IterableDataset shuffle
        if take_docs:
            ds = ds.take(take_docs)

        self.ds = ds
        self.block_size = block_size
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc.eot_token if eos else None
        self.max_tokens = max_tokens
        self.seed = seed

    def _shard_for_workers(self):
        info = get_worker_info()
        if info is None:
            return self.ds
        # Deterministic per-worker shard
        return self.ds.shard(num_shards=info.num_workers, index=info.id)

    def __iter__(self):
        rng = random.Random(self.seed)
        ds = self._shard_for_workers()
        token_buf = collections.deque()
        emitted = 0

        for ex in ds:
            txt = ex["text"] if "text" in ex else (ex.get("content") or "")
            if not txt:
                continue
            ids = self.enc.encode_ordinary(txt)
            if self.eot is not None:
                ids.append(self.eot)
            token_buf.extend(ids)

            # Slice out contiguous, NON-overlapping blocks
            while len(token_buf) >= self.block_size + 1:
                # Take T+1 tokens so that:
                #   tokens[:-1] is x, tokens[1:] is y
                tok = list(itertools.islice(token_buf, 0, self.block_size + 1))
                # Drop what we consumed (advance by T, not T+1, to keep doc boundary via overlap of 1)
                for _ in range(self.block_size):
                    token_buf.popleft()

                tokens = torch.tensor(tok, dtype=torch.long)  # shape [T+1]
                emitted += len(tokens)
                yield tokens

                if self.max_tokens and emitted >= self.max_tokens:
                    return

def get_dataloader(batch_size=8, num_workers=4, **kw):
    ds = FineWebPacked(**kw)
    def collate(batch):
        return torch.stack(batch, dim=0)  # shape [B, T+1]
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      pin_memory=True, collate_fn=collate)
