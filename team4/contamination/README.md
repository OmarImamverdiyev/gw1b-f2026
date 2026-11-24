# Team 4 — Pretraining contamination scan (scan A)

Checks every question of the seven suite benchmarks against all of Team 1's pretraining shards
(272 files, 26.72B tokens) and reports which questions appear in the training data.

**Rule.** A question is *flagged* if any run of 13 tokens in a row (a 13-gram) from the question
also appears in the shards. The share of the question's 13-grams found (`frac_matched`) is
reported too, so a stricter threshold can be applied later without rescanning. Overlap is a
warning, not proof of memorization.

The test index and the 13-gram hash are Seth's (`team4/rq42_sft/overlap/`), the same ones the
SFT scan uses, so the team has one definition of overlap and one index.

## Files

| File | What it does |
| --- | --- |
| `scan_shards.py` | Scans one slice of the shards (`--part K/N`); writes `hits.partK.npz` and `summary.partK.json` |
| `merge_flags.py` | Checks every part finished with the same index, then writes `flags.jsonl`, `summary.json`, `top_matches.txt` |
| `test_scan_shards.py` | 10 tests on fake shards in the real file format: planted copy, chunk edge, clean shard, short question, brute-force agreement, dropped hashes, refusals, positive control passes / fails on a tokenizer mismatch |
| `planted_check.py` | Positive control on REAL data: plants two real benchmark questions in a copy of a real shard; the scan must find them |
| `submit_scan.sh` | Submits the N parts as `gw1b run --cpu` jobs |

## Steps (run from the root of your clone of the team fork, on Pegasus)

```bash
export C=$GW1B_SCRATCH/users/$USER/contamination
TOK=<tokenizer .model path from Team 1's manifest.json>
TASKS=<the seven suite tasks, comma-separated, exactly as lm-eval names them>
```

**1. Build the test index** (Seth's builder; CPU job; downloads benchmark data):
```bash
gw1b run --cpu --cpus 4 --mem 16G --time 2:00:00 --name contam-index \
  team4/rq42_sft/overlap/build_test_index.py --tokenizer "$TOK" --lm-eval-tasks "$TASKS" --out $C/index
```
Check the job output: it prints the number of items, hashes, and short items per task.

**2. Run the tests** (seconds; CPU job, never on the login node):
```bash
gw1b run --cpu --time 0:20:00 --name contam-tests -m pytest -q team4/contamination/test_scan_shards.py
```
Expect `10 passed`.

**2b. Positive control on real data** (must pass before the full scan; a few minutes of CPU):
```bash
SHARD1=$(ls /scratch/gw1b-class/teams/team1/data/fineweb-edu-100BT-34f-bpe32k/train/*.bin | head -1)
gw1b run --cpu --cpus 2 --mem 8G --time 0:30:00 --name contam-plant team4/contamination/planted_check.py \
  make --tokenizer "$TOK" --shard "$SHARD1" --out $C/planted
```
When it finishes, scan and merge that one file, then check:
```bash
gw1b run --cpu --cpus 2 --mem 4G --time 0:20:00 --name contam-plant-scan team4/contamination/scan_shards.py \
  --index $C/index --shards "$C/planted/planted_000000.bin" --out $C/planted/hits
# after that job ends:
gw1b run --cpu --cpus 2 --mem 8G --time 0:20:00 --name contam-plant-merge team4/contamination/merge_flags.py \
  --index $C/index --hits $C/planted/hits --out $C/planted/merged
# after that job ends (reads two small files):
gw1b run --cpu --time 0:10:00 --name contam-plant-check team4/contamination/planted_check.py \
  check --planted $C/planted/planted.json --flags $C/planted/merged/flags.jsonl
```
Expect two `PASS` lines and `POSITIVE CONTROL PASSED`. A `FAIL` means the index and the shards
disagree (tokenizer, text extraction or task names): stop and fix that first, because the full
scan would report "clean" without being able to see anything.

**3. Time one real shard:**
```bash
gw1b run --cpu --cpus 2 --mem 4G --time 0:30:00 --name contam-timing team4/contamination/scan_shards.py \
  --index $C/index --shards "/scratch/gw1b-class/teams/team1/data/fineweb-edu-100BT-34f-bpe32k/train/*.bin" \
  --out $C/timing --limit-files 1
```
Off-cluster this took ~21 s and ~0.7 GB for a 98M-token shard. If Pegasus is much slower, raise `NPARTS`.

**4. Submit the full scan** (8 jobs by default, ~12 min each at the measured speed):
```bash
INDEX=$C/index OUT=$C/hits bash team4/contamination/submit_scan.sh
squeue -u $USER
```

**5. Merge** once every part has finished:
```bash
gw1b run --cpu --cpus 2 --mem 16G --time 0:30:00 --name contam-merge team4/contamination/merge_flags.py \
  --index $C/index --hits $C/hits --out $C/merged --tokenizer "$TOK"
```

**6. Review stock phrasing.** Open `$C/merged/top_matches.txt`. If a frequent 13-gram is generic
wording rather than a leaked question, put its hash (third column) in `$C/drop.txt`, one per
line, write down why, and rerun step 5 with `--drop-hashes $C/drop.txt`.

## Outputs

`flags.jsonl`: one line per question, in lm-eval order, so it joins RQ 4.1's per-question records on `(task, doc_id)`:
```json
{"task": "arc_easy", "doc_id": 17, "n_ngrams": 24, "matched_ngrams": 3, "frac_matched": 0.125,
 "n_hits": 5, "flagged": true, "short": false, "shards": ["train_000041.bin"],
 "example": {"shard": "train_000041.bin", "pos": 81234567}}
```
`summary.json`: per benchmark: items, short (under 13 tokens, undetectable), flagged, flagged at ≥50% of 13-grams, flagged %.

## Limits to report

- Questions under 13 tokens cannot be detected (`short` in the summary).
- The production run trains on 23B of the 26.72B tokens; `shards` in each flag allows narrowing to shards actually trained on.
- Copies with different wording, spacing or formatting are missed: the rule needs 13 identical tokens in a row.
- Documents are packed with an end-of-text token between them, so a 13-gram can span two documents; a chance match of 13 tokens is very unlikely.
