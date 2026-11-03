"""Compare LR-sweep runs only when their other experimental settings match."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+")
    a = p.parse_args()
    results, reference = [], None
    for name in a.runs:
        run = Path(name)
        cfg = json.loads((run / "sft_config.json").read_text())
        summary = json.loads((run / "summary.json").read_text())
        if not summary["finished"] or summary["step"] != cfg["steps"]:
            raise ValueError(f"Incomplete run: {run}")
        invariant = {key: cfg[key] for key in ("base_checkpoints", "base_step_loaded", "steps", "seq_len",
                    "micro_batch", "grad_accum", "warmup", "loss_chunk", "seed", "template",
                    "base_config_sha256", "versions", "code_sha256")}
        invariant["data_hashes"] = {s: cfg["data_manifest"]["files"][s]["sha256"] for s in ("train", "val")}
        invariant["tokenizer"] = cfg["data_manifest"]["tokenizer_sha256prefix"]
        if reference is not None and invariant != reference:
            raise ValueError(f"Changed settings other than LR: {run}")
        reference = invariant
        results.append({"run": str(run), "lr": cfg["lr"], "held_out_loss": summary["val_loss"],
                        "gpu_hours": summary["gpu_hours"]})
    if len({r["lr"] for r in results}) != len(results):
        raise ValueError("Duplicate LR; compare one fixed seed at a time")
    results.sort(key=lambda r: r["held_out_loss"])
    print(json.dumps({"selection_metric": "held-out instruction response-token loss",
                      "selected_lr": results[0]["lr"], "results": results}, indent=2))


if __name__ == "__main__":
    main()
