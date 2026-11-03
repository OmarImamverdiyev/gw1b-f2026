"""Response-only supervised fine-tuning using the shared GW1B JAX toolchain.

Single device prototype. Each run starts from base weights, not optimizer state.
"""
import argparse
import dataclasses
import functools
import hashlib
import json
import math
import os
from importlib.metadata import version
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from gw1b import checkpoint
from gw1b.config import load_config, save_config
from gw1b.model import count_params
from gw1b.sharding import create_model, make_mesh
from gw1b.tokenizer import Tokenizer
from gw1b.train import build_optimizer
from chat_template import TEMPLATE_VERSION
from sft_data import BatchStream, collate, encode_rows, prompt_key, read_jsonl


@functools.partial(nnx.jit, static_argnames=("loss_chunk", "grad_accum"))
def train_step(model, optimizer, batch, loss_chunk=64, grad_accum=1):
    """Accumulate gradients weighted by response tokens, then take ONE Adam step."""
    micro = batch["inputs"].shape[0] // grad_accum
    total_tokens = jnp.maximum(batch["mask"].sum(), 1)
    total_loss, total_grads = jnp.float32(0), None
    for i in range(grad_accum):
        b = {k: v[i * micro:(i + 1) * micro] for k, v in batch.items()}
        weight = b["mask"].sum() / total_tokens

        def objective(m):
            return m.loss(b["inputs"], b["targets"], b["mask"], chunk=loss_chunk) * weight

        loss, grads = nnx.value_and_grad(objective)(model)
        total_loss += loss
        total_grads = grads if total_grads is None else jax.tree.map(jnp.add, total_grads, grads)
    grad_norm = optax.global_norm(total_grads)
    optimizer.update(model, total_grads)
    return total_loss, grad_norm


@functools.partial(nnx.jit, static_argnames=("loss_chunk",))
def eval_step(model, batch, loss_chunk=64):
    return model.loss(batch["inputs"], batch["targets"], batch["mask"], chunk=loss_chunk)


def evaluate(model, rows, batch_size, loss_chunk):
    nll = tokens = 0
    for i in range(0, len(rows), batch_size):
        batch = collate(rows[i:i + batch_size])
        n = int(batch["mask"].sum())
        value = float(eval_step(model, {k: jnp.asarray(v) for k, v in batch.items()}, loss_chunk))
        nll += value * n
        tokens += n
    return nll / tokens


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-config", required=True)
    p.add_argument("--base-checkpoints", required=True, help="Manager root, ending in /checkpoints")
    p.add_argument("--base-step", type=int, required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--data", required=True, help="Directory containing train.jsonl, val.jsonl, manifest.json")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup", type=int, default=5, help="Smoke=5; research runs start with 100")
    p.add_argument("--loss-chunk", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-cpu", action="store_true", help="Local integration tests only")
    a = p.parse_args(argv)
    if min(a.steps, a.seq_len, a.micro_batch, a.grad_accum, a.loss_chunk,
           a.eval_every, a.save_every) < 1 or a.lr <= 0 or not 0 <= a.warmup < a.steps:
        p.error("Invalid sizes, learning rate or warmup")
    if a.seq_len % a.loss_chunk:
        p.error("loss-chunk must divide seq-len to keep memory bounded")
    if os.environ.get("SLURM_JOB_ID") is None and not a.allow_cpu:
        p.error("Run through gw1b run on a compute node")
    if jax.device_count() != 1:
        p.error("This prototype supports one device; multi-device scaling is not implemented")
    if jax.default_backend() != "gpu" and not a.allow_cpu:
        p.error("GPU required (local tests can use --allow-cpu)")
    if a.allow_cpu and os.environ.get("SLURM_JOB_ID") is None and os.uname().nodename.startswith("log"):
        p.error("Do not train on a Pegasus login node")
    data = Path(a.data).resolve()
    manifest = json.loads((data / "manifest.json").read_text())
    tok = Tokenizer(a.tokenizer)
    if manifest["template"] != TEMPLATE_VERSION or manifest["seq_len"] != a.seq_len:
        raise ValueError("Prepared data template/context mismatch")
    if manifest["tokenizer_sha256prefix"] != tok.sha256:
        raise ValueError("Prepared data tokenizer mismatch")
    raw = {}
    for split in ("train", "val"):
        path = data / f"{split}.jsonl"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest["files"][split]["sha256"]:
            raise ValueError(f"Changed {split} data; regenerate manifest")
        raw[split] = read_jsonl(path)
    train_keys = {prompt_key(r["prompt"], r.get("system", "")) for r in raw["train"]}
    val_keys = {prompt_key(r["prompt"], r.get("system", "")) for r in raw["val"]}
    if train_keys & val_keys:
        raise ValueError("Train/validation instruction overlap")
    train_rows, val_rows = (encode_rows(raw[s], tok, a.seq_len) for s in ("train", "val"))
    base = load_config(a.base_config)
    if tok.vocab_size != base.model.vocab_size or a.seq_len > base.model.max_seq_len:
        raise ValueError("Tokenizer vocabulary/context does not match base architecture")
    # Metadata guards against accidentally pairing a same-sized, different tokenizer.
    base_meta = checkpoint.read_meta(a.base_checkpoints, a.base_step)
    base_tokenizer = base_meta.get("config", {}).get("data", {}).get("tokenizer")
    expected_tokenizer = base_tokenizer or base.data.tokenizer
    if Tokenizer(os.path.expandvars(expected_tokenizer)).sha256 != tok.sha256:
        raise ValueError("Tokenizer differs from the checkpoint's saved tokenizer")
    run = Path(a.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=False)  # Never overwrite or implicitly resume a previous SFT run.
    cfg = dataclasses.replace(
        base,
        model=dataclasses.replace(base.model, dtype="auto"),
        data=dataclasses.replace(base.data, tokenizer=str(Path(a.tokenizer).resolve()), seq_len=a.seq_len),
        optim=dataclasses.replace(base.optim, name="adamw", lr=a.lr, warmup_steps=a.warmup,
                                  schedule="cosine", min_lr_ratio=0.1, grad_accum=None),
        run=dataclasses.replace(base.run, name=run.name, out_dir=str(run.parent),
                                seed=a.seed, total_steps=a.steps, total_tokens=None,
                                batch_size=a.micro_batch * a.grad_accum, shard_params=False,
                                loss_chunk=a.loss_chunk, ckpt_every=a.save_every, tensorboard=False))
    started = time.monotonic()
    mesh = make_mesh(1)
    with jax.set_mesh(mesh):
        model = create_model(cfg, mesh)
        loaded_step = checkpoint.load_model_only(a.base_checkpoints, model, a.base_step)
        tx, schedule = build_optimizer(cfg.optim, a.steps)
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        if int(optimizer.step[...]) != 0:
            raise AssertionError("SFT must start at optimizer step zero")
        cfg = dataclasses.replace(cfg, model=model.cfg)
        save_config(cfg, str(run / "config.yaml"))
        settings = vars(a) | {"base_step_loaded": loaded_step, "optimizer_initial_step": 0,
                              "template": TEMPLATE_VERSION, "data_manifest": manifest,
                              "device": str(jax.devices()[0]), "n_params": count_params(model),
                              "versions": {k: version(k) for k in ("jax", "flax", "optax", "orbax-checkpoint")},
                              "code_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                              for f in Path(__file__).parent.glob("*.py")},
                              "base_config_sha256": hashlib.sha256(Path(a.base_config).read_bytes()).hexdigest()}
        (run / "sft_config.json").write_text(json.dumps(settings, indent=2) + "\n")
        stream = BatchStream(train_rows, a.seed)
        probe = train_rows[:min(64, len(train_rows))]
        tokens_seen = response_tokens = step = 0
        last_loss = None
        initial_train = evaluate(model, probe, a.micro_batch, a.loss_chunk)
        initial_val = evaluate(model, val_rows, a.micro_batch, a.loss_chunk)
        print(json.dumps({"step": 0, "train_probe_loss": initial_train, "val_loss": initial_val}), flush=True)
        mgr = checkpoint.make_manager(str(run / "checkpoints"), a.save_every, keep=2, async_save=False)
        finished = False
        log = (run / "metrics.jsonl").open("w", buffering=1)

        def metadata():
            return {"step": step, "tokens_seen": tokens_seen, "response_tokens_seen": response_tokens,
                    "train_seconds": time.monotonic() - started, "train_loss": last_loss,
                    "val_loss": val_loss, "config": cfg.to_dict(), "sft": True,
                    "base_checkpoint": a.base_checkpoints, "base_step": loaded_step,
                    "template": TEMPLATE_VERSION}

        val_loss = initial_val
        try:
            log.write(json.dumps({"step": 0, "train_probe_loss": initial_train, "val_loss": initial_val}) + "\n")
            for step in range(1, a.steps + 1):
                batch = stream.take(cfg.run.batch_size)
                response_tokens += int(batch["mask"].sum())
                tokens_seen += int((batch["inputs"] != tok.pad_id).sum())
                loss, norm = train_step(model, optimizer, {k: jnp.asarray(v) for k, v in batch.items()},
                                        loss_chunk=a.loss_chunk, grad_accum=a.grad_accum)
                last_loss, grad_norm = float(loss), float(norm)
                if not math.isfinite(last_loss) or not math.isfinite(grad_norm):
                    raise FloatingPointError(f"Non-finite training at step {step}")
                metrics = {"step": step, "train_loss": last_loss, "grad_norm": grad_norm,
                           "lr": float(schedule(step - 1)), "tokens_seen": tokens_seen,
                           "response_tokens_seen": response_tokens,
                           "epochs": stream.examples_seen / len(train_rows),
                           "gpu_hours": (time.monotonic() - started) / 3600 if jax.default_backend() == "gpu" else 0}
                if step % a.eval_every == 0 or step == a.steps:
                    val_loss = evaluate(model, val_rows, a.micro_batch, a.loss_chunk)
                    metrics["val_loss"] = val_loss
                    print(json.dumps(metrics), flush=True)
                log.write(json.dumps(metrics) + "\n")
                if step % a.save_every == 0 or step == a.steps:
                    checkpoint.save(mgr, step, model, optimizer, metadata(), force=True)
            finished = True
        except KeyboardInterrupt:
            print("Interrupted; saving completed optimizer steps", flush=True)
            step = int(optimizer.step[...])
            if step:
                checkpoint.save(mgr, step, model, optimizer, metadata(), force=True)
        finally:
            log.close()
            mgr.close()
        final_train = evaluate(model, probe, a.micro_batch, a.loss_chunk)
        final_val = evaluate(model, val_rows, a.micro_batch, a.loss_chunk)
        summary = metadata() | {"finished": finished, "n_params": count_params(model),
                  "optimizer_final_step": int(optimizer.step[...]),
                  "train_probe_loss_before": initial_train, "train_probe_loss_after": final_train,
                  "val_loss_before": initial_val, "val_loss": final_val,
                  "train_probe_loss_fell": final_train < initial_train,
                  "epochs": stream.examples_seen / len(train_rows), "truncation_rate": 0.0,
                  "seed": a.seed, "gpu_hours": (time.monotonic() - started) / 3600 if jax.default_backend() == "gpu" else 0,
                  "estimated_train_flops": 6 * count_params(model) * step * cfg.run.batch_size * a.seq_len,
                  "flops_note": "6*N*padded training positions estimate; excludes eval, checkpointing and compile",
                  "purpose": "engineering smoke test; unscanned data, not H4/H5 evidence"}
        checkpoint.write_run_summary(str(run), summary)
        print(json.dumps(summary, indent=2), flush=True)
        if not finished:
            raise SystemExit(130)


if __name__ == "__main__":
    main()
