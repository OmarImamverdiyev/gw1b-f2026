#!/usr/bin/env bash
# Team 3, E1b: peak learning rate at 100M on V100 (float32), AdamW, 2B tokens each (20 tokens/parameter).
#
#   bash team3/experiments/e1b_lr_100m.sh <center_lr> [weight_decay]      # 3 runs: 0.5x, 1x, 2x center
#   ... add --submit to queue.
#
# center_lr: the 100M value extrapolated from E1's 50M optimum; weight_decay: E1's best AdamW value (default 0.1).
# The winner (with the 50M optimum) is the tuned learning rate per size delivered to Teams 1 and 2.
source "$(dirname "$0")/_common.sh"
[[ $# -ge 1 ]] || { echo "usage: $0 <center_lr> [weight_decay] [--submit]" >&2; exit 2; }
CENTER="$1"; WD="${2:-0.1}"
COMMON="--gpus 4 --gpu-type v100 --time 1-12:00:00 --config configs/team3/e1b_100m.yaml"
for f in 0.5 1 2; do
  lr=$(python3 -c "m, e = f'{$CENTER*$f:.2e}'.split('e'); print(m.rstrip('0').rstrip('.') + 'e' + str(int(e)))")
  emit $COMMON --set run.name="team3-e1b-adamw-lr$lr-wd$WD-s0" --set optim.name=adamw \
    --set optim.lr="$lr" --set optim.weight_decay="$WD" $(seed_args 0)
done
finish
