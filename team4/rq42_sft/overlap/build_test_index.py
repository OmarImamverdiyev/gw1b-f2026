"""Step 1: hash every 13-gram of the TEST sets (IFEval + 7 benchmarks, later GW set + safety prompts).

  gw1b run --cpu --cpus 4 --mem 16G --time 1:00:00 overlap/build_test_index.py \
      --tokenizer /SEAS/groups/gw1b-class/tokenizer/gw1b-32k.model --out $T4/data/overlap/test_index \
      --lm-eval-tasks ifeval,hellaswag,arc_easy,arc_challenge,piqa,winogrande,lambada_openai,mmlu
  # later (Nov 21): --items-jsonl gw_heldout_v1.jsonl safety_prompts.jsonl   rows: {"task","doc_id","text"}

Only the dataset's OWN words are hashed (lm-eval templates like "Question:"/"Answer:" are stripped).
Items shorter than 13 tokens can never match; they are counted in test_items.json["short"].
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from ngram import N, ngram_hashes, test_variants

TEMPLATE_NOISE = re.compile(r"^(Question:\s*)|(\s*Answer:\s*)$")


def flatten(d, out=None):
    out = {} if out is None else out
    for k, v in d.items():
        flatten(v, out) if isinstance(v, dict) else out.__setitem__(getattr(k, "group_name", k) if not isinstance(k, str) else k, v)
    return out


def lm_eval_items(task_names):
    from lm_eval.tasks import TaskManager, get_task_dict
    for name, task in flatten(get_task_dict(task_names, TaskManager())).items():
        if not hasattr(task, "eval_docs"):
            continue
        for doc_id, doc in enumerate(task.eval_docs):  # SAME order lm-eval / RQ 4.1 use
            try:
                if name == "ifeval":
                    text = doc["prompt"]
                else:
                    text = task.doc_to_text(doc)
                    choices = task.doc_to_choice(doc) if task.config.doc_to_choice else None
                    target = task.doc_to_target(doc)
                    if isinstance(text, int):          # winogrande: text is a choice index
                        text = choices[text]
                    if isinstance(target, int) and choices:
                        target = choices[target]
                    text = TEMPLATE_NOISE.sub("", str(text)) + " " + str(target)
            except Exception as e:  # never silently drop a test item
                raise RuntimeError(f"Cannot extract text for {name}[{doc_id}]; cross-check with Yusra") from e
            yield {"task": name, "doc_id": doc_id, "text": text}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lm-eval-tasks", default="")
    p.add_argument("--items-jsonl", nargs="*", default=[])
    a = p.parse_args()
    from gw1b.tokenizer import Tokenizer
    tok = Tokenizer(a.tokenizer)
    items = list(lm_eval_items(a.lm_eval_tasks.split(","))) if a.lm_eval_tasks else []
    for f in a.items_jsonl:
        items += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    build(items, tok, Path(a.out))


def build(items, tok, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    hashes, owners, meta, short = [], [], [], Counter()
    for i, it in enumerate(items):
        variants = test_variants(tok, it["text"])
        # An item under 13 tokens can never match; don't let the leading-space variant (+1 token) change that.
        hs = [ngram_hashes(t) for t in variants] if len(variants[0]) >= N else [np.zeros(0, np.uint64)]
        u = np.unique(np.concatenate(hs))
        hashes.append(u); owners.append(np.full(len(u), i, dtype=np.int32))
        meta.append({"task": it["task"], "doc_id": it["doc_id"], "n_ngrams": int(len(u))})
        if len(u) == 0:
            short[it["task"]] += 1
    H = np.concatenate(hashes) if hashes else np.zeros(0, np.uint64)
    O = np.concatenate(owners) if owners else np.zeros(0, np.int32)
    order = np.argsort(H, kind="stable")
    np.savez(out / "index.npz", hashes=H[order], owners=O[order])
    (out / "test_items.json").write_text(json.dumps(
        {"n": N, "items": meta, "short_items_per_task": dict(short),
         "total_items": len(meta), "total_hashes": int(len(H))}, indent=1))
    print(f"{len(meta)} test items, {len(H)} hashes, short (<{N} tokens): {dict(short)}")


if __name__ == "__main__":
    main()
