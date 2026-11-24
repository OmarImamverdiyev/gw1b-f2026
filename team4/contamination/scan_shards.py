"""Scan pretraining token shards for 13-token runs shared with the benchmark test index.

Team 4 contamination scan A: benchmarks vs. Team 1's pretraining shards.

The test index comes from team4/rq42_sft/overlap/build_test_index.py (the same index and the
same 13-gram hash as the SFT scan, so the team has one definition of "overlap").

Each job scans one slice of the shard list (--part K/N takes every N-th file starting at K)
and writes:
    <out>/hits.partK.npz      every match: hash, file index, token position
    <out>/summary.partK.json  files scanned, tokens per file, match count, index fingerprint

Shards are memory-mapped and read in chunks of --chunk tokens. Each chunk starts N-1 = 12
tokens before the previous one ended, so a copy that straddles a chunk edge is still found,
and no 13-gram is counted twice.

Run merge_flags.py after every part has finished.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

OVERLAP_DIR = Path(__file__).resolve().parent.parent / "rq42_sft" / "overlap"
sys.path.insert(0, str(OVERLAP_DIR))
from ngram import N, ngram_hashes  # noqa: E402  (Seth's 13-gram hash: one rule for every scan)

from gw1b.data import read_shard  # noqa: E402  (class shard reader: 1024-byte header + uint16 ids)


def load_index(index_dir: Path):
    """Sorted unique hashes of every test 13-gram, plus the index file's SHA-256 fingerprint."""
    path = index_dir / "index.npz"
    z = np.load(path)
    uniq = np.unique(z["hashes"])  # index.npz is sorted but may repeat a hash owned by several items
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return uniq, sha


def select_files(patterns: list[str], part: int, nparts: int) -> tuple[list[str], list[int]]:
    """All shard files (sorted, absolute) and the indices this part scans."""
    files = sorted({str(Path(f).resolve()) for pat in patterns for f in glob.glob(pat)})
    if not files:
        raise SystemExit(f"No shard files match {patterns}")
    if not 0 <= part < nparts:
        raise SystemExit(f"--part {part}/{nparts} is out of range")
    return files, list(range(part, len(files), nparts))


def scan_tokens(tokens: np.ndarray, uniq: np.ndarray, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (hashes, positions) of every 13-gram in `tokens` that is in `uniq`.

    A position is the index of the 13-gram's first token in `tokens`.
    """
    n = len(tokens)
    found_h, found_p = [], []
    for start in range(0, n, chunk):
        lo = max(0, start - (N - 1))          # 12-token overlap with the previous chunk
        window = np.asarray(tokens[lo:start + chunk])
        h = ngram_hashes(window)
        if len(h) == 0 or len(uniq) == 0:
            continue
        idx = np.searchsorted(uniq, h)        # binary search in the sorted index
        idx[idx == len(uniq)] = 0
        hit = uniq[idx] == h
        if hit.any():
            where = np.flatnonzero(hit)
            found_h.append(h[where])
            found_p.append(lo + where.astype(np.int64))
    if not found_h:
        return np.zeros(0, np.uint64), np.zeros(0, np.int64)
    return np.concatenate(found_h), np.concatenate(found_p)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", required=True, help="folder with index.npz and test_items.json")
    p.add_argument("--shards", nargs="+", required=True, help="shard file glob(s), e.g. '.../train/*.bin'")
    p.add_argument("--out", required=True, help="results folder (shared by all parts)")
    p.add_argument("--part", default="0/1", help="K/N: scan every N-th file starting at K")
    p.add_argument("--chunk", type=int, default=10_000_000, help="tokens per chunk (memory ~ 30 bytes x chunk)")
    p.add_argument("--limit-files", type=int, default=0, help="scan only the first L files of this part (timing)")
    a = p.parse_args(argv)
    if a.chunk < N:
        p.error(f"--chunk must be at least {N}")

    part, nparts = map(int, a.part.split("/"))
    files, mine = select_files(a.shards, part, nparts)
    if a.limit_files:
        mine = mine[:a.limit_files]
    uniq, index_sha = load_index(Path(a.index))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    all_h, all_f, all_p, per_file = [], [], [], []
    for i in mine:
        t1 = time.time()
        tokens = read_shard(files[i])
        h, pos = scan_tokens(tokens, uniq, a.chunk)
        all_h.append(h)
        all_p.append(pos)
        all_f.append(np.full(len(h), i, dtype=np.int32))
        per_file.append({"file": i, "tokens": int(len(tokens)), "matches": int(len(h)),
                         "seconds": round(time.time() - t1, 1)})
        print(f"part {part}/{nparts}: {Path(files[i]).name} {len(tokens):,} tokens, "
              f"{len(h):,} matches, {time.time() - t1:.1f}s", flush=True)

    np.savez(out / f"hits.part{part}.npz",
             hashes=np.concatenate(all_h) if all_h else np.zeros(0, np.uint64),
             files=np.concatenate(all_f) if all_f else np.zeros(0, np.int32),
             positions=np.concatenate(all_p) if all_p else np.zeros(0, np.int64))
    summary = {"part": part, "nparts": nparts, "n": N, "chunk": a.chunk,
               "index_sha256": index_sha, "index_unique_hashes": int(len(uniq)),
               "all_files": files, "scanned": per_file, "complete": not a.limit_files,
               "tokens": sum(f["tokens"] for f in per_file),
               "matches": sum(f["matches"] for f in per_file),
               "seconds": round(time.time() - t0, 1)}
    (out / f"summary.part{part}.json").write_text(json.dumps(summary, indent=1))
    print(f"part {part}/{nparts} done: {len(per_file)} files, {summary['tokens']:,} tokens, "
          f"{summary['matches']:,} matches, {summary['seconds']}s")


if __name__ == "__main__":
    main()
