# Shared helpers for the Team 3 launch scripts. Sourced, not run.
# Every script prints its `gw1b train` commands; add --submit to actually queue them.
set -euo pipefail

SUBMIT=0
ARGS=()
for a in "$@"; do
  if [[ "$a" == "--submit" ]]; then SUBMIT=1; else ARGS+=("$a"); fi
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

# Paired seeds: model init (run.seed) and data order (data.seed) move together.
seed_args() { echo "--set run.seed=$1 --set data.seed=$1"; }

emit() {  # emit <gw1b train args...>
  if [[ $SUBMIT -eq 1 ]]; then
    echo "+ gw1b train $*"
    # shellcheck disable=SC2068
    gw1b train $@
  else
    echo "gw1b train $*"
  fi
}

finish() {
  if [[ $SUBMIT -eq 0 ]]; then
    echo
    echo "# dry run: nothing submitted. Re-run with --submit to queue these jobs."
  fi
}
