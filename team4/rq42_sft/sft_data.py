"""Small JSONL instruction loader, independent of the pretraining shard split."""
import hashlib
import json
import random
import re
import numpy as np

from chat_template import encode_example


def prompt_key(prompt: str, system: str = "") -> str:
    # Same instruction cannot enter both train and validation under a new answer.
    canonical = re.sub(r"\s+", " ", system.strip() + "\n" + prompt.strip()).casefold()
    return hashlib.sha256(canonical.encode()).hexdigest()


def single_turn(row: dict) -> dict | None:
    messages = row.get("messages", [])
    system = ""
    if messages and messages[0].get("role") == "system":
        system = messages[0].get("content", "")
        messages = messages[1:]
    # Keep the first exchange only: later turns could require dropped context.
    if len(messages) < 2 or [m.get("role") for m in messages[:2]] != ["user", "assistant"]:
        return None
    prompt, response = (m.get("content") for m in messages[:2])
    if not all(isinstance(x, str) for x in (system, prompt, response)):
        return None
    if not prompt.strip() or not response.strip():
        return None
    return {"prompt": prompt.strip(), "response": response.strip(), "system": system.strip()}


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f"Empty instruction split: {path}")
    return rows


def encode_rows(rows, tokenizer, seq_len):
    encoded = []
    for i, row in enumerate(rows):
        try:
            encoded.append(encode_example(tokenizer, row["prompt"], row["response"],
                                          seq_len, row.get("system", "")))
        except (ValueError, OverflowError) as e:
            raise ValueError(f"Invalid row {i}; prepare data for this context first") from e
    return encoded


def collate(rows):
    return {k: np.asarray([r[k] for r in rows], dtype=np.float32 if k == "mask" else np.int32)
            for k in ("inputs", "targets", "mask")}


class BatchStream:
    """Deterministic epoch shuffles; no dropped tail examples."""
    def __init__(self, rows, seed):
        if not rows:
            raise ValueError("No training rows")
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
        return collate(batch)
