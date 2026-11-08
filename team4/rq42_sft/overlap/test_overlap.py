import numpy as np
from pathlib import Path
from ngram import N, ngram_hashes
import build_test_index as bti
import scan_pool as sp


class CharTok:
    def encode(self, text, add_bos=False, add_eos=False):
        return [10 + ord(c) for c in text]


Q = ["What is the capital city of the country France? Paris",
     "Which gas do plants absorb from the air for photosynthesis? Carbon dioxide",
     "A farmer has 12 apples and gives away 5 of them. How many remain? 7",
     "Short q? yes"]


def build(tmp):
    items = [{"task": "arc_easy", "doc_id": i, "text": q} for i, q in enumerate(Q)]
    bti.build(items, CharTok(), tmp)
    z = np.load(tmp / "index.npz")
    import json
    return z["hashes"], z["owners"], json.loads((tmp / "test_items.json").read_text())


def test_flags_exactly_planted(tmp_path):
    H, O, info = build(tmp_path)
    assert info["short_items_per_task"] == {"arc_easy": 1}
    rows = [("a:0", "src", {"prompt": "Tell me: " + Q[0] + " thanks", "response": "ok"}),
            ("a:1", "src", {"prompt": "hi", "response": "Sure. " + Q[1]}),
            ("a:2", "src2", {"messages": [{"role": "user", "content": "x"},
                                          {"role": "assistant", "content": "  " + Q[2] + "\n"}]}),
            ("a:3", "src", {"prompt": Q[3] + " " + Q[3], "response": "short test items never match"}),
            ("a:4", "src", {"prompt": Q[0][:12], "response": Q[0][30:42]}),     # only 12 tok shared
            ("a:5", "src", {"prompt": "unrelated chat about cooking pasta tonight", "response": "boil water"}),
            ("a:6", "src", {"prompt": Q[0][:12], "response": Q[0][12:24]})]
    flagged, counts, totals = sp.scan_rows(rows, CharTok(), H, O, info["items"], sp.group_of)
    assert [r["id"] for r in flagged] == ["a:0", "a:1", "a:2"]
    assert flagged[0]["fields"] == ["prompt"] and flagged[1]["fields"] == ["response"]
    assert flagged[0]["longest_run"] >= len(Q[0]) - 1 and "arc_easy:0" in flagged[0]["test_items"]
    assert counts["src"]["arc_easy"] == 2 and counts["src2"]["arc_easy"] == 1


def test_hash_matches_playbook_formula():
    t = np.arange(20, dtype=np.uint64)
    h = np.zeros(8, dtype=np.uint64)
    for k in range(N):
        h = h * np.uint64(1_000_003) + t[k:k + 8]
    np.testing.assert_array_equal(ngram_hashes(t), h)
