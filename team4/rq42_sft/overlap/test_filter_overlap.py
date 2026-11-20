"""Integration: build index -> scan (2 shards) -> merge -> filter -> manifest. No gw1b/GPU needed."""
import hashlib, json, sys, types
import pytest
import build_test_index as bti
import scan_pool as sp
import filter_overlap as fo
from test_overlap import CharTok, Q


class FakeTok(CharTok):
    def __init__(self, path=None): pass


@pytest.fixture
def env(tmp_path, monkeypatch):
    mod = types.ModuleType("gw1b.tokenizer"); mod.Tokenizer = FakeTok
    monkeypatch.setitem(sys.modules, "gw1b", types.ModuleType("gw1b"))
    monkeypatch.setitem(sys.modules, "gw1b.tokenizer", mod)
    bti.build([{"task": "arc_easy", "doc_id": i, "text": q} for i, q in enumerate(Q)], CharTok(), tmp_path / "idx")
    data = tmp_path / "data"; data.mkdir()
    def w(name, rows):
        # blank line first: IDs must count physical lines like scan_pool.read_pool
        (data / name).write_text("\n" + "".join(json.dumps(r) + "\n" for r in rows))
    w("train.jsonl", [{"prompt": "hi " + Q[0], "response": "ok", "source": "A"},
                      {"prompt": "cook pasta", "response": "boil water", "source": "A"},
                      {"prompt": "x", "response": Q[1], "source": "B"},
                      {"prompt": "weather", "response": "sunny today", "source": "B"}])
    w("val.jsonl", [{"prompt": "greet", "response": "hello there friend", "source": "A"},
                    {"prompt": "sum", "response": "Easy: " + Q[2], "source": "B"}])
    files = {s: {"path": f"{s}.jsonl", "rows": 0, "sha256": "old", "sources": {}} for s in ("train", "val")}
    (data / "manifest.json").write_text(json.dumps({"template": "t", "seq_len": 64, "files": files, "purpose": "smoke"}))
    return tmp_path, data


def run_scan(tmp, data, nshards=2, only=None):
    for s in range(nshards):
        if only is not None and s not in only: continue
        sys.argv = ["scan_pool.py", "--tokenizer", "x", "--index", str(tmp / "idx"), "--pool",
                    str(data / "train.jsonl"), str(data / "val.jsonl"), "--out", str(tmp / "scan"), "--shard", f"{s}/{nshards}"]
        sp.main()


def test_end_to_end(env):
    tmp, data = env
    run_scan(tmp, data)
    sp.merge(tmp / "scan")
    out = tmp / "clean"
    sys.argv = ["f", "--scan", str(tmp / "scan"), "--data", str(data), "--out", str(out)]
    fo.main()
    train = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines()]
    val = [json.loads(l) for l in (out / "val.jsonl").read_text().splitlines()]
    assert [r["prompt"] for r in train] == ["cook pasta", "weather"]
    assert [r["prompt"] for r in val] == ["greet"]
    m = json.loads((out / "manifest.json").read_text())
    assert m["overlap"]["removed"] == {"train": 2, "val": 1} and m["overlap"]["test_sets"] == ["arc_easy"]
    assert m["files"]["train"]["rows"] == 2 and m["files"]["train"]["sources"] == {"A": 1, "B": 1}
    for s in ("train", "val"):
        assert m["files"][s]["sha256"] == hashlib.sha256((out / f"{s}.jsonl").read_bytes()).hexdigest()
    assert json.loads((tmp / "scan" / "overlap_summary.json").read_text())["examples_flagged"] == 3


def test_merge_rejects_missing_shard(env):
    tmp, data = env
    run_scan(tmp, data, only=[0])
    with pytest.raises(SystemExit):
        sp.merge(tmp / "scan")


def test_filter_rejects_wrong_files(env):
    tmp, data = env
    run_scan(tmp, data); sp.merge(tmp / "scan")
    other = tmp / "other"; other.mkdir()
    for n in ("train.jsonl", "val.jsonl", "manifest.json"):
        (other / n).write_text((data / n).read_text())
    sys.argv = ["f", "--scan", str(tmp / "scan"), "--data", str(other), "--out", str(tmp / "c2")]
    with pytest.raises(SystemExit):
        fo.main()
