"""Preference-pair loader for DPO. Shares the SFT chat template and prompt hashing.

Chosen and rejected responses answer the same prompt, so both sides are encoded with
`chat_template.encode_example` and differ only in their response tokens and mask.
"""
import json
import random
import numpy as np

from chat_template import encode_example
from sft_data import prompt_key

SIDES = ("chosen", "rejected")


def preference_row(row: dict) -> dict | None:
    """Normalise one preference record; return None when it carries no usable signal."""
    system = (row.get("system") or "").strip()
    prompt = (row.get("prompt") or "").strip()
    chosen = (row.get("chosen") or "").strip()
    rejected = (row.get("rejected") or "").strip()
    if not prompt or not chosen or not rejected:
        return None
    # A tie contributes a constant zero margin and therefore no gradient; drop it explicitly
    # rather than letting it dilute the batch mean.
    if chosen == rejected:
        return None
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected, "system": system}


def read_pairs(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]
    rows = [r for r in (preference_row(x) for x in raw) if r is not None]
    if not rows:
        raise ValueError(f"No usable preference pairs: {path}")
    return rows


def encode_pairs(rows, tokenizer, seq_len):
    """Encode both sides of every pair; reject overflow rather than truncating an answer."""
    encoded = []
    for i, row in enumerate(rows):
        try:
            sides = {s: encode_example(tokenizer, row["prompt"], row[s], seq_len, row["system"])
                     for s in SIDES}
        except (ValueError, OverflowError) as e:
            raise ValueError(f"Invalid preference row {i}; prepare pairs for this context first") from e
        encoded.append(sides)
    return encoded


def collate_pairs(rows):
    """Stack both sides into one array of 2P sequences: chosen first, then rejected.

    One forward pass then covers both sides, and `split_sides` recovers them. Keeping the
    order fixed is what lets the loss pair row i with row i + P.
    """
    out = {}
    for key in ("inputs", "targets", "mask"):
        dtype = np.float32 if key == "mask" else np.int32
        out[key] = np.asarray([r[s][key] for s in SIDES for r in rows], dtype=dtype)
    return out


def split_sides(values):
    """Split a [2P, ...] array produced by `collate_pairs` into (chosen, rejected)."""
    pairs = values.shape[0] // 2
    return values[:pairs], values[pairs:]


class PairStream:
    """Deterministic epoch shuffles over preference pairs; no dropped tail examples."""

    def __init__(self, rows, seed):
        if not rows:
            raise ValueError("No preference pairs")
        self.rows, self.rng = rows, random.Random(seed)
        self.order = list(range(len(rows)))
        self.rng.shuffle(self.order)
        self.position = self.examples_seen = 0

    def take(self, n):
        batch = []
        for _ in range(n):
            if self.position == len(self.order):
                self.rng.shuffle(self.order)
                self.position = 0
            batch.append(self.rows[self.order[self.position]])
            self.position += 1
            self.examples_seen += 1
        return collate_pairs(batch)


def pair_keys(rows) -> set[str]:
    """Prompt hashes, so the same instruction cannot sit in both train and validation."""
    return {prompt_key(r["prompt"], r.get("system", "")) for r in rows}
