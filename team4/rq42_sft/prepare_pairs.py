"""Fetch a bounded preference-pair sample on a Slurm CPU node; pin source revisions.

This is NOT H5 evidence. The pairs come from a public preference set, so they are
off-policy by construction: the responses were written by other models. H5's SFT-sampled
condition needs pairs generated from our own SFT checkpoint, which `sample.py` produces
and this script deliberately does not fake.

No benchmark or safety prompts enter training; run the overlap scan over `chosen` and
`rejected` as well as the prompt before any run that is meant as evidence.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random

from chat_template import TEMPLATE_VERSION, encode_example
from dpo_data import preference_row
from sft_data import prompt_key

# (repo, subset, split). Both ship `prompt`, `chosen` and `rejected` as message lists.
SOURCES = [("HuggingFaceH4/ultrafeedback_binarized", "default", "train_prefs"),
           ("allenai/llama-3.1-tulu-3-8b-preference-mixture", "default", "train")]


def last_assistant(messages) -> str:
    """Final assistant turn of a chosen/rejected conversation, or '' if there is none."""
    if isinstance(messages, str):
        return messages.strip()
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return (message.get("content") or "").strip()
    return ""


def first_user(messages) -> tuple[str, str]:
    """(system, first user turn) of a conversation, or ('', '') if there is none."""
    if not isinstance(messages, list):
        return "", ""
    system = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role, content = message.get("role"), (message.get("content") or "").strip()
        if role == "system" and not system:
            system = content
        elif role == "user":
            return system, content
    return system, ""


def preference_turn(row: dict) -> dict | None:
    """Normalise one upstream row into {prompt, chosen, rejected, system}.

    Handles both shapes we pull: a bare `prompt` string alongside message lists, and rows
    where the prompt only exists inside the conversations. Multi-turn pairs are dropped
    rather than flattened, matching `single_turn`'s rule for SFT.
    """
    chosen_messages, rejected_messages = row.get("chosen"), row.get("rejected")
    system, prompt = first_user(chosen_messages)
    prompt = (row.get("prompt") or prompt or "").strip()
    chosen, rejected = last_assistant(chosen_messages), last_assistant(rejected_messages)
    if isinstance(chosen_messages, list) and len([m for m in chosen_messages
                                                  if isinstance(m, dict) and m.get("role") == "user"]) > 1:
        return None  # later turns could depend on context this template drops
    return preference_row({"prompt": prompt, "chosen": chosen, "rejected": rejected, "system": system})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--val-size", type=int, default=64)
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
    for source_index, (repo, subset, split) in enumerate(SOURCES):
        info = HfApi().dataset_info(repo)
        revision = info.sha
        stream = load_dataset(repo, subset, split=split, revision=revision, streaming=True)
        stream = stream.shuffle(seed=a.seed, buffer_size=1000)
        source_count = 0
        # Equal source quotas for this development sample, not a research filter.
        quota = wanted // len(SOURCES) + (source_index < wanted % len(SOURCES))
        for row in itertools.islice(stream, a.max_rows_per_source):
            count["examined"] += 1
            record = preference_turn(row)
            if record is None:
                count["invalid_or_tied"] += 1
                continue
            key = prompt_key(record["prompt"], record["system"])
            if key in seen:
                count["duplicate_prompt"] += 1
                continue
            if sum(len(v) for v in record.values()) > 20000:
                count["overlong_char_guard"] += 1
                continue
            try:
                # Both sides must fit; a pair whose rejected answer overflows is not a pair.
                for side in ("chosen", "rejected"):
                    encode_example(tok, record["prompt"], record[side], a.seq_len, record["system"])
            except OverflowError:
                count["over_context"] += 1
                continue
            record.update(source=repo, subset=subset, split=split, revision=revision,
                          pair_source="off-policy-public-dataset", prompt_sha256=key)
            records.append(record)
            seen.add(key)
            source_count += 1
            if source_count >= quota:
                break
        if source_count != quota:
            raise RuntimeError(f"Only {source_count}/{quota} usable pairs from {repo}; increase scan bound")
        sources.append({"dataset": repo, "subset": subset, "split": split, "revision": revision,
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
    manifest = {"purpose": "DPO engineering smoke test only; off-policy pairs, overlap scan pending",
                "sources": sources, "seed": a.seed, "seq_len": a.seq_len,
                "template": TEMPLATE_VERSION, "tokenizer_sha256prefix": tok.sha256,
                "counts": dict(count), "truncated": 0, "files": files,
                "pair_source": "off-policy-public-dataset",
                "split_rule": "Deduplicate normalized system+prompt, seeded shuffle, validation first",
                "conversation_rule": "Single-turn pairs only; retain initial system message",
                "h5_note": ("Off-policy by construction. The SFT-sampled arm of H5 must be generated from "
                            "our own SFT checkpoint over these same prompts, with the same pair count and "
                            "labelling rule, so only the response source differs."),
                "license_note": "Review linked dataset cards and constituent terms before redistribution"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
