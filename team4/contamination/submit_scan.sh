#!/usr/bin/env bash
# Submit the pretraining contamination scan as NPARTS CPU jobs (one slice of the shards each).
#
# Run from the root of your clone of the team fork, on a Pegasus login node:
#     INDEX=$GW1B_SCRATCH/users/$USER/contamination/index \
#     OUT=$GW1B_SCRATCH/users/$USER/contamination/hits \
#     bash team4/contamination/submit_scan.sh
#
# Uses `gw1b run --cpu` (not plain sbatch) so every job gets the class environment.
# Measured on a full-size fake shard (98M tokens): ~21 s and ~0.7 GB per shard on one core,
# so 8 parts x 34 shards is ~12 min per job. --time is set well above that.
set -euo pipefail

SHARDS="${SHARDS:-/scratch/gw1b-class/teams/team1/data/fineweb-edu-100BT-34f-bpe32k/train/*.bin}"
NPARTS="${NPARTS:-8}"
TIME="${TIME:-2:00:00}"
: "${INDEX:?set INDEX to the folder with index.npz and test_items.json}"
: "${OUT:?set OUT to the folder the parts write to}"

[ -f "$INDEX/index.npz" ] || { echo "No $INDEX/index.npz: build the index first (README step 1)"; exit 1; }
n=$(compgen -G "$SHARDS" | wc -l)
[ "$n" -gt 0 ] || { echo "No shard files match $SHARDS"; exit 1; }
if compgen -G "$OUT/summary.part*.json" > /dev/null; then
    echo "$OUT already has results; use a new OUT folder so old and new parts never mix"; exit 1
fi
mkdir -p "$OUT"
echo "Submitting $NPARTS parts over $n shard files -> $OUT"

for ((k = 0; k < NPARTS; k++)); do
    gw1b run --cpu --cpus 2 --mem 4G --time "$TIME" --name "contam-$k" \
        team4/contamination/scan_shards.py \
        --index "$INDEX" --shards "$SHARDS" --out "$OUT" --part "$k/$NPARTS"
done
echo "Watch with: squeue -u \$USER   (gw1b status is broken; see the troubleshooting log)"
echo "When all parts are done, run merge_flags.py (README step 6)."
