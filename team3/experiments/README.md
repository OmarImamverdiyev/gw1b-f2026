# Team 3 launch scripts

Each script prints the `gw1b train` commands it would run; add `--submit` to queue them. Run them from a
checkout of this branch on Pegasus, activated so that jobs use this branch's code (Muon, `run.eval_seqs`)
rather than the shared class copy:

```bash
cp -n /SEAS/groups/gw1b-class/gw1b-f2026/gw1b.local.env <checkout>/   # once: site settings (git-ignored)
source <checkout>/activate.sh                                       # each session
cd <checkout>
```

Setting `PYTHONPATH` alone is not enough: the class copy is always placed first.

| Script | Experiment | Runs |
|---|---|---|
| `e0_pilot.sh smoke\|adamw\|resume\|muon` | E0: tiny end-to-end run; 50M AdamW pilot (200 steps); resume to 300 steps; 50M Muon pilot | 1 each |
| `estimate_cost.py <adamw pilot> --muon <muon pilot>` | Prices E1 and E1b from the pilots' measured throughput | none |
| `e1_tune_50m.sh stage1` | E1: AdamW and Muon, 5 learning rates at weight decay 0.1 | 10 |
| `e1_tune_50m.sh stage2 <adamw lrs> <muon lrs>` | E1: weight decay 0.3 and 1.0 at each optimizer's best 3 learning rates | 12 |
| `e1_tune_50m.sh seeds <adamw lr> <wd> <muon lr> <wd>` | E1: seeds 1 and 2 of each optimizer's winner | 4 |
| `e1b_lr_100m.sh <center lr> [wd]` | E1b: AdamW at 100M, 0.5x / 1x / 2x the value extrapolated from 50M | 3 |

Order: `e0_pilot.sh` (all four stages) → `estimate_cost.py` → E1 stage1 → stage2 → seeds → E1b.

Conventions used by every run:

- V100s, 4 GPUs per job, float32 for both optimizers (`configs/team3/*.yaml`).
- Every run is scored on the same 512 validation windows (`run.eval_seqs=512`).
- Seeds are paired: `run.seed` (initialization) and `data.seed` (data order) take the same value.
- Weight decay uses the optax convention (multiplied by the learning rate).
- Muon uses update-RMS matching (`optim.muon_consistent_rms=0.2`), so it shares AdamW's learning-rate grid.
- Run names: `team3-<experiment>-<optimizer>-lr<lr>-wd<wd>-s<seed>`, in `$GW1B_SCRATCH/teams/team3/runs`.
