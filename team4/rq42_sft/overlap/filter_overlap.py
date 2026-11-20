"""Step 3: apply a MERGED overlap scan (scan_pool.py --merge) and write a cleaned copy of the data.

Prepared SFT dir (train.jsonl, val.jsonl, manifest.json from prepare_data.py). The scan must have been run
with --pool <dir>/train.jsonl <dir>/val.jsonl:
  python overlap/filter_overlap.py --scan $T4/data/overlap/scan_nov7 --data $T4/data/sft-smoke --out $T4/data/sft-smoke-clean

Raw pool files (any JSONL; rows kept byte-for-byte, no manifest):
  python overlap/filter_overlap.py --scan SCAN --pool raw/smoltalk.jsonl raw/tulu3.jsonl --out raw_clean

Safety checks: the files filtered must be exactly the files scanned, every removal ID must be found,
and the scanned-row count must equal the rows read. IDs are numbered exactly as scan_pool.read_pool does.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def rows_with_ids(path):
    """Yield (id, raw_line, row). line_no counts ALL physical lines, like scan_pool.read_pool."""
    stem = Path(path).stem
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line.strip():
                yield f"{stem}:{line_no}", line if line.endswith("\n") else line + "\n", json.loads(line)


def filter_file(path, out_path, remove):
    kept, removed, by_source, found = 0, 0, Counter(), set()
    with open(out_path, "w", encoding="utf-8") as out:
        for ex_id, raw, row in rows_with_ids(path):
            if ex_id in remove:
                removed += 1
                found.add(ex_id)
                by_source[str(row.get("source") or row.get("original_source") or Path(path).stem)] += 1
            else:
                out.write(raw)
                kept += 1
    return kept, removed, by_source, found


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scan", required=True, help="merged scan dir (remove_ids.txt, overlap_summary.json)")
    p.add_argument("--out", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--data", help="prepared SFT dir with train.jsonl, val.jsonl, manifest.json")
    g.add_argument("--pool", nargs="+", help="raw JSONL files that were scanned")
    a = p.parse_args()

    scan = Path(a.scan)
    summary = json.loads((scan / "overlap_summary.json").read_text())
    remove = set((scan / "remove_ids.txt").read_text().splitlines())
    remove.discard("")
    if a.data:
        data = Path(a.data).resolve()
        files = [data / "train.jsonl", data / "val.jsonl"]
    else:
        files = [Path(f).resolve() for f in a.pool]
    scanned = {str(Path(f).resolve()) for f in summary["files"]}
    if scanned != {str(f) for f in files}:
        raise SystemExit(f"Scan covered {sorted(scanned)}, but filtering {[str(f) for f in files]}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)

    stats, found = {}, set()
    for f in files:
        kept, removed, by_source, ids = filter_file(f, out / f.name, remove)
        stats[f.stem] = {"kept": kept, "removed": removed, "removed_by_source": dict(by_source)}
        found |= ids
    total_read = sum(s["kept"] + s["removed"] for s in stats.values())
    if total_read != summary["examples_scanned"]:
        raise SystemExit(f"Read {total_read} rows but the scan covered {summary['examples_scanned']}: stale scan?")
    if found != remove:
        raise SystemExit(f"{len(remove - found)} removal IDs not found in these files (e.g. {sorted(remove - found)[:3]})")

    overlap = {"rule": summary["rule"], "n": summary["n"], "index_sha256": summary["index_sha256"],
               "test_sets": summary["test_sets"], "scanned": summary["examples_scanned"],
               "removed": {k: v["removed"] for k, v in stats.items()},
               "removed_by_source": {k: v["removed_by_source"] for k, v in stats.items()},
               "flagged_by_source_and_test_set": summary["flagged_by_source_and_test_set"]}
    if a.data:
        manifest = json.loads((data / "manifest.json").read_text())
        for split in ("train", "val"):
            if stats[split]["kept"] == 0:
                raise SystemExit(f"Cleaned {split} split is empty")
            path = out / f"{split}.jsonl"
            manifest["files"][split].update(
                rows=stats[split]["kept"], sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                sources=dict(Counter(json.loads(l).get("source") for l in path.open(encoding="utf-8") if l.strip())))
        manifest["overlap"] = overlap
        manifest["purpose"] = "SFT data after test-set overlap removal; see 'overlap.test_sets' for what was checked"
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        (out / "overlap.json").write_text(json.dumps(overlap, indent=2) + "\n")
    print(json.dumps(overlap, indent=2))


if __name__ == "__main__":
    main()
