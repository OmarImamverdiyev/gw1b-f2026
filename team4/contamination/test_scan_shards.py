"""Tests for the pretraining contamination scan (scan_shards.py + merge_flags.py).

Run before touching real data:  python -m pytest -q team4/contamination/test_scan_shards.py

Fake shards are written with the class write_shard (the real on-disk format) and the index is
built with Seth's build() from team4/rq42_sft/overlap, so the whole path matches production.
A fake tokenizer reads text as space-separated token ids, so a test controls every token.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rq42_sft" / "overlap"))
sys.path.insert(0, str(HERE.parent.parent))  # repo root, for the gw1b package

from build_test_index import build  # noqa: E402
from gw1b.data import write_shard  # noqa: E402
from ngram import N  # noqa: E402
import merge_flags  # noqa: E402
import scan_shards  # noqa: E402


class FakeTok:
    """'5 6 7' -> [5, 6, 7]."""
    def encode(self, text):
        return [int(w) for w in text.split()]


def text(ids):
    return " ".join(str(int(i)) for i in ids)


def make_index(tmp, questions):
    """questions: list of (task, token list). Returns the index folder."""
    items = [{"task": t, "doc_id": i, "text": text(q)} for i, (t, q) in enumerate(questions)]
    d = tmp / "index"
    build(items, FakeTok(), d)
    return d


def make_shards(tmp, arrays):
    d = tmp / "shards"
    d.mkdir(exist_ok=True)
    for k, a in enumerate(arrays):
        write_shard(str(d / f"train_{k:06d}.bin"), np.asarray(a, dtype=np.uint16))
    return str(d / "*.bin")


def run(tmp, index, shards, nparts=1, chunk=10_000_000, drop=""):
    hits = tmp / "hits"
    for k in range(nparts):
        scan_shards.main(["--index", str(index), "--shards", shards, "--out", str(hits),
                          "--part", f"{k}/{nparts}", "--chunk", str(chunk)])
    out = tmp / "merged"
    args = ["--index", str(index), "--hits", str(hits), "--out", str(out)]
    if drop:
        args += ["--drop-hashes", drop]
    merge_flags.main(args)
    rows = [json.loads(l) for l in open(out / "flags.jsonl")]
    return rows, json.loads((out / "summary.json").read_text())


def noise(rng, n):
    return rng.integers(0, 900, size=n)        # never collides with ids >= 1000 used for questions


def test_planted_copy_is_found_at_its_position(tmp_path):
    rng = np.random.default_rng(0)
    q = np.arange(1000, 1020)                  # 20 tokens -> 8 thirteen-grams
    shard = noise(rng, 2000)
    shard[500:515] = q[3:18]                   # 15 copied tokens -> 3 thirteen-grams
    rows, summary = run(tmp_path, make_index(tmp_path, [("arc_easy", q)]), make_shards(tmp_path, [shard]))
    r = rows[0]
    assert r["flagged"] and r["matched_ngrams"] == 3 and r["n_hits"] == 3
    assert r["n_ngrams"] == 8 and r["frac_matched"] == 0.375
    assert r["example"]["pos"] in (500, 501, 502)
    assert summary["per_task"]["arc_easy"]["flagged"] == 1


def test_copy_across_a_chunk_edge_is_found(tmp_path):
    rng = np.random.default_rng(1)
    q = np.arange(1000, 1030)
    shard = noise(rng, 300)
    shard[58:78] = q[:20]                      # straddles the chunk edge at 64
    uniq, _ = scan_shards.load_index(make_index(tmp_path, [("piqa", q)]))
    small_h, small_p = scan_shards.scan_tokens(shard.astype(np.uint16), uniq, chunk=64)
    big_h, big_p = scan_shards.scan_tokens(shard.astype(np.uint16), uniq, chunk=10_000)
    assert sorted(small_p.tolist()) == sorted(big_p.tolist()) == list(range(58, 66))
    assert len(small_p) == len(set(small_p.tolist()))     # nothing counted twice


def test_a_clean_shard_gives_no_flags(tmp_path):
    rng = np.random.default_rng(2)
    rows, summary = run(tmp_path, make_index(tmp_path, [("sciq", np.arange(1000, 1040))]),
                        make_shards(tmp_path, [noise(rng, 5000)]))
    assert not rows[0]["flagged"] and summary["matches_total"] == 0


def test_short_question_is_reported_not_flagged(tmp_path):
    rng = np.random.default_rng(3)
    q = np.arange(1000, 1010)                  # 10 tokens: can never match
    shard = noise(rng, 500)
    shard[100:110] = q
    rows, summary = run(tmp_path, make_index(tmp_path, [("copa", q)]), make_shards(tmp_path, [shard]))
    assert rows[0]["short"] and not rows[0]["flagged"]
    assert summary["per_task"]["copa"]["short"] == 1


def test_matches_brute_force_on_random_data(tmp_path):
    """Small vocabulary -> many chance matches; flags must equal a plain set comparison."""
    rng = np.random.default_rng(4)
    shards = [rng.integers(0, 3, size=n) for n in (4000, 3500, 5000, 2500)]
    questions = []
    for i in range(40):
        if i % 3 == 0:                          # copied from a shard
            s = shards[i % 4]; st = int(rng.integers(0, len(s) - 30))
            questions.append((f"task{i % 2}", s[st:st + 30]))
        else:
            questions.append((f"task{i % 2}", rng.integers(0, 3, size=int(rng.integers(5, 40)))))
    rows, _ = run(tmp_path, make_index(tmp_path, questions), make_shards(tmp_path, shards),
                  nparts=3, chunk=257)
    corpus = Counter13()
    for s in shards:
        corpus.add(s)
    for (task, q), r in zip(questions, rows):
        grams = {tuple(q[k:k + N]) for k in range(len(q) - N + 1)}
        present = [g for g in grams if g in corpus.counts]
        assert r["task"] == task
        assert r["matched_ngrams"] == len(present)
        assert r["n_hits"] == sum(corpus.counts[g] for g in present)
        assert r["flagged"] == bool(present)


class Counter13:
    def __init__(self):
        self.counts = {}

    def add(self, s):
        for k in range(len(s) - N + 1):
            g = tuple(s[k:k + N])
            self.counts[g] = self.counts.get(g, 0) + 1


def test_dropped_hash_removes_the_flag(tmp_path):
    rng = np.random.default_rng(5)
    q = np.arange(1000, 1013)                  # exactly one 13-gram
    shard = noise(rng, 400)
    shard[50:63] = q
    index = make_index(tmp_path, [("mmlu", q)])
    rows, _ = run(tmp_path, index, make_shards(tmp_path, [shard]))
    assert rows[0]["flagged"]
    drop = tmp_path / "drop.txt"
    drop.write_text(str(int(np.load(index / "index.npz")["hashes"][0])))
    rows, summary = run(tmp_path, index, make_shards(tmp_path, [shard]), drop=str(drop))
    assert not rows[0]["flagged"] and summary["dropped_hashes"] == 1


def test_merge_refuses_a_missing_part(tmp_path):
    rng = np.random.default_rng(6)
    index = make_index(tmp_path, [("hellaswag", np.arange(1000, 1020))])
    shards = make_shards(tmp_path, [noise(rng, 300), noise(rng, 300)])
    scan_shards.main(["--index", str(index), "--shards", shards, "--out", str(tmp_path / "h"), "--part", "0/2"])
    with pytest.raises(SystemExit, match="Missing or mixed parts"):
        merge_flags.main(["--index", str(index), "--hits", str(tmp_path / "h"), "--out", str(tmp_path / "m")])


def test_merge_refuses_a_timing_run(tmp_path):
    rng = np.random.default_rng(7)
    index = make_index(tmp_path, [("hellaswag", np.arange(1000, 1020))])
    shards = make_shards(tmp_path, [noise(rng, 300), noise(rng, 300)])
    scan_shards.main(["--index", str(index), "--shards", shards, "--out", str(tmp_path / "h"),
                      "--limit-files", "1"])
    with pytest.raises(SystemExit, match="timing run"):
        merge_flags.main(["--index", str(index), "--hits", str(tmp_path / "h"), "--out", str(tmp_path / "m")])


# --- positive control (planted_check.py) ----------------------------------------------------

import planted_check  # noqa: E402

EOS = 2


def planted_setup(tmp, index_shift=0):
    """Plant two fake questions the way planted_check does; build the index from the same
    questions (shifted ids = a tokenizer that disagrees with the shards)."""
    rng = np.random.default_rng(8)
    base = rng.integers(3, 900, size=5000)
    base[rng.choice(5000, 40, replace=False)] = EOS          # document boundaries
    q_mid, q_start = np.arange(1000, 1030), np.arange(2000, 2025)
    docs = [[5, 6, 7] + q_mid.tolist() + [8, 9, EOS], q_start.tolist() + [EOS]]
    tokens, starts = planted_check.plant(base, docs, EOS)
    assert tokens[starts[0]:starts[0] + len(docs[0])].tolist() == docs[0]
    assert tokens[starts[1] - 1] == EOS and tokens[starts[0] - 1] == EOS   # planted at boundaries
    index = make_index(tmp, [("arc_easy", q_mid + index_shift), ("arc_easy", q_start + index_shift)])
    planted = tmp / "planted.json"
    planted.write_text(json.dumps({"planted": [
        {"task": "arc_easy", "doc_id": 0, "style": "mid"}, {"task": "arc_easy", "doc_id": 1, "style": "start"}]}))
    run(tmp, index, make_shards(tmp, [tokens]))
    return str(planted), str(tmp / "merged" / "flags.jsonl")


def test_positive_control_passes_when_pieces_match(tmp_path):
    assert planted_check.check(*planted_setup(tmp_path))


def test_positive_control_fails_on_a_tokenizer_mismatch(tmp_path):
    assert not planted_check.check(*planted_setup(tmp_path, index_shift=5000))
