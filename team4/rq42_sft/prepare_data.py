"""Fetch a bounded development sample on a Slurm CPU node; pin source revisions.

This is NOT the cleaned H4 data tier. No benchmark or safety prompts enter training.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random

from chat_template import TEMPLATE_VERSION, encode_example
from sft_data import prompt_key, single_turn

SOURCES = [("HuggingFaceTB/smoltalk", "all"), ("allenai/tulu-3-sft-mixture", "default")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--train-size", type=int, default=1000)
    p.add_argument("--val-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows-per-source", type=int, default=30000)
    a = p.parse_args()
    if min(a.train_size, a.val_size, a.seq_len, a.max_rows_per_source) < 1:
        p.error("Sizes must be positive")
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from gw1b.tokenizer import Tokenizer
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    tok = Tokenizer(a.tokenizer)
    count, records, seen, sources = Counter(), [], set(), []
    wanted = a.train_size + a.val_size
    for source_index, (repo, subset) in enumerate(SOURCES):
        info = HfApi().dataset_info(repo)
        revision = info.sha
        stream = load_dataset(repo, subset, split="train", revision=revision, streaming=True)
        stream = stream.shuffle(seed=a.seed, buffer_size=1000)
        source_count = 0
        # Equal source quotas for this development sample, not a research filter.
        quota = wanted // len(SOURCES) + (source_index < wanted % len(SOURCES))
        for row in itertools.islice(stream, a.max_rows_per_source):
            count["examined"] += 1
            record = single_turn(row)
            if record is None:
                count["invalid"] += 1
                continue
            key = prompt_key(record["prompt"], record["system"])
            if key in seen:
                count["duplicate_prompt"] += 1
                continue
            if sum(len(v) for v in record.values()) > 20000:
                count["overlong_char_guard"] += 1
                continue
            try:
                encode_example(tok, record["prompt"], record["response"], a.seq_len, record["system"])
            except OverflowError:
                count["over_context"] += 1
                continue
            record.update(source=repo, subset=subset, revision=revision,
                          original_source=str(row.get("source", "")), prompt_sha256=key)
            records.append(record)
            seen.add(key)
            source_count += 1
            if source_count >= quota:
                break
        if source_count != quota:
            raise RuntimeError(f"Only {source_count}/{quota} usable rows from {repo}; increase scan bound")
        sources.append({"dataset": repo, "subset": subset, "split": "train", "revision": revision,
                        "accepted": source_count, "card": f"https://huggingface.co/datasets/{repo}"})
    random.Random(a.seed).shuffle(records)
    files = {}
    for name, rows in [("val", records[:a.val_size]), ("train", records[a.val_size:])]:
        path = out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        files[name] = {"path": path.name, "rows": len(rows),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "sources": dict(Counter(r["source"] for r in rows))}
    manifest = {"purpose": "SFT engineering smoke test only; overlap scan pending",
                "sources": sources, "seed": a.seed, "seq_len": a.seq_len,
                "template": TEMPLATE_VERSION, "tokenizer_sha256prefix": tok.sha256,
                "counts": dict(count), "truncated": 0, "files": files,
                "split_rule": "Deduplicate normalized system+prompt, seeded shuffle, validation first",
                "conversation_rule": "First user/assistant exchange; retain initial system message",
                "license_note": "Review linked dataset cards and constituent terms before redistribution"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
