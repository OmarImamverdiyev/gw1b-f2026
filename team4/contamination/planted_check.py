"""Positive control for the pretraining scan: prove the real pieces fit together.

The unit tests use a fake tokenizer and fake shards. This check uses the REAL tokenizer, a REAL
benchmark question (extracted exactly as the index extracts it) and a REAL training shard, and
fails loudly if the scanner cannot find a question we know is there. Without it, "0 flagged"
could mean "clean data" or "blind scanner", and nobody could tell which.

    make   copy the first --tokens tokens of a real shard and plant two copies of benchmark
           questions in it, as Team 1's pipeline would have stored them (plain text + end-of-text
           token, no start token):
             * "mid":   inside a sentence, the way a question sits in a web page
             * "start": as a document of its own
           Writes <out>/planted_000000.bin and <out>/planted.json (what was planted, where).
    check  read the merged flags.jsonl of a scan of that one file and print PASS/FAIL per question.

Run make, then scan_shards.py + merge_flags.py on the planted shard only, then check (README).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

OVERLAP_DIR = Path(__file__).resolve().parent.parent / "rq42_sft" / "overlap"
sys.path.insert(0, str(OVERLAP_DIR))
from ngram import normalize  # noqa: E402  (same whitespace cleaning the index applies)

PASS_FRAC = 0.8   # a planted copy should match nearly all of the question's 13-grams


def wrap(text: str, style: str) -> str:
    if style == "mid":
        return f"Here is a practice question from a science worksheet. {text} Students should explain their reasoning."
    return text


def plant(base: np.ndarray, docs: list[list[int]], eos_id: int) -> tuple[np.ndarray, list[int]]:
    """Insert each tokenized document right after an end-of-text token, spread through `base`.

    Returns the new token array and where each document starts. Inserting at a document
    boundary mirrors how Team 1 packed documents back to back.
    """
    base = np.asarray(base)
    eos = np.flatnonzero(base == eos_id)
    out, starts, prev = [], [], 0
    for k, doc in enumerate(docs):
        target = (k + 1) * len(base) // (len(docs) + 1)
        later = eos[eos >= max(target, prev)]
        cut = int(later[0]) + 1 if len(later) else max(target, prev)
        out.append(base[prev:cut])
        starts.append(sum(len(x) for x in out))
        out.append(np.asarray(doc, dtype=base.dtype))
        prev = cut
    out.append(base[prev:])
    return np.concatenate(out), starts


def load_question(task: str, doc_id: int) -> str:
    from build_test_index import lm_eval_items   # Seth's extractor: identical text to the index
    for it in lm_eval_items([task]):
        if it["task"] == task and it["doc_id"] == doc_id:
            return it["text"]
    raise SystemExit(f"{task}[{doc_id}] not found")


def cmd_make(a) -> None:
    from gw1b.data import read_shard, write_shard
    from gw1b.tokenizer import Tokenizer
    tok = Tokenizer(a.tokenizer)
    base = np.asarray(read_shard(a.shard)[:a.tokens])
    planted, docs = [], []
    for style, doc_id in (("mid", a.doc_ids[0]), ("start", a.doc_ids[1])):
        text = normalize(load_question(a.task, doc_id))
        ids = tok.encode(wrap(text, style), add_eos=True)        # as gw1b.prepare_data does
        docs.append(ids)
        planted.append({"task": a.task, "doc_id": doc_id, "style": style, "text": text,
                        "doc_tokens": len(ids)})
    tokens, starts = plant(base, docs, tok.eos_id)
    for p, s in zip(planted, starts):
        p["position"] = int(s)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_shard(str(out / "planted_000000.bin"), tokens.astype(base.dtype))
    (out / "planted.json").write_text(json.dumps(
        {"base_shard": str(Path(a.shard).resolve()), "base_tokens": int(len(base)),
         "total_tokens": int(len(tokens)), "planted": planted}, indent=1))
    for p in planted:
        print(f"planted {p['task']}[{p['doc_id']}] ({p['style']}) at token {p['position']}: {p['text'][:80]!r}")
    print(f"wrote {out}/planted_000000.bin ({len(tokens):,} tokens) and planted.json")


def check(planted_json: str, flags_jsonl: str) -> bool:
    planted = json.loads(Path(planted_json).read_text())["planted"]
    rows = {(r["task"], r["doc_id"]): r for r in map(json.loads, open(flags_jsonl))}
    ok = True
    for p in planted:
        r = rows.get((p["task"], p["doc_id"]))
        if r is None:
            print(f"FAIL {p['task']}[{p['doc_id']}] ({p['style']}): not in flags.jsonl (wrong index?)")
            ok = False
        elif r["short"]:
            print(f"FAIL {p['task']}[{p['doc_id']}] ({p['style']}): under 13 tokens; pick another --doc-ids")
            ok = False
        elif not r["flagged"] or r["frac_matched"] < PASS_FRAC:
            print(f"FAIL {p['task']}[{p['doc_id']}] ({p['style']}): flagged={r['flagged']}, "
                  f"matched {r['matched_ngrams']}/{r['n_ngrams']} 13-grams")
            ok = False
        else:
            print(f"PASS {p['task']}[{p['doc_id']}] ({p['style']}): matched {r['matched_ngrams']}/{r['n_ngrams']} "
                  f"13-grams ({r['frac_matched']:.0%})")
    print("POSITIVE CONTROL PASSED" if ok else "POSITIVE CONTROL FAILED: do not trust the full scan yet")
    return ok


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--tokenizer", required=True)
    m.add_argument("--shard", required=True, help="one real training shard to copy from")
    m.add_argument("--out", required=True)
    m.add_argument("--tokens", type=int, default=2_000_000, help="how much of the shard to copy")
    m.add_argument("--task", default="arc_easy")
    m.add_argument("--doc-ids", type=int, nargs=2, default=[0, 1], metavar=("MID", "START"))
    c = sub.add_parser("check")
    c.add_argument("--planted", required=True, help="planted.json written by make")
    c.add_argument("--flags", required=True, help="flags.jsonl from merging a scan of the planted shard")
    a = ap.parse_args(argv)
    if a.cmd == "make":
        cmd_make(a)
    elif not check(a.planted, a.flags):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
