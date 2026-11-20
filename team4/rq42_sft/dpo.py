"""Direct Preference Optimization on top of an SFT checkpoint, using the shared GW1B toolchain.

Single device prototype, mirroring sft.py. Policy and frozen reference both start from the
same SFT checkpoint, so the implicit reward is zero at step 0 and every later change is
attributable to this run. The reference is never updated and never receives gradients.

Loss (Rafailov et al. 2023):
    r(x, y)   = beta * (log pi(y|x) - log pi_ref(y|x))      summed over response tokens
    L         = -log sigmoid(r(x, y_chosen) - r(x, y_rejected))
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
from dpo_data import PairStream, collate_pairs, encode_pairs, pair_keys, read_pairs, split_sides

# optax.global_norm is deprecated in favour of optax.tree.norm (Robel, #team4 2026-11-11).
_global_norm = getattr(getattr(optax, "tree", None), "norm", None) or optax.global_norm


def sequence_logprob(model, inputs, targets, mask, chunk: int = 64):
    """Sum of log p(target | prefix) over masked positions, per sequence. Returns [B].

    DPO compares whole responses, so unlike `model.loss` this is a per-sequence sum and not a
    batch mean. Chunking mirrors `gw1b.model.chunked_cross_entropy`: one chunk of [B, chunk, V]
    float32 log-probabilities is live at a time, forward and backward.
    """
    x = model.hidden(inputs)                       # [B, T, d]
    w = model.output_matrix()                      # [V, d]
    B, T, _ = x.shape
    mask = mask.astype(jnp.float32)
    if chunk <= 0 or T % chunk or T == chunk:
        logits = jnp.einsum("btd,vd->btv", x, w, preferred_element_type=jnp.float32)
        logp = jax.nn.log_softmax(logits, axis=-1)
        token_logp = jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
        return (token_logp * mask).sum(axis=-1)
    n = T // chunk
    xs = x.reshape(B, n, chunk, -1).transpose(1, 0, 2, 3)        # [n, B, chunk, d]
    ts = targets.reshape(B, n, chunk).transpose(1, 0, 2)         # [n, B, chunk]
    ms = mask.reshape(B, n, chunk).transpose(1, 0, 2)            # [n, B, chunk]

    @jax.checkpoint
    def one(carry, inputs_):
        x_c, t_c, m_c = inputs_
        logits = jnp.einsum("bcd,vd->bcv", x_c, w, preferred_element_type=jnp.float32)
        logp = jax.nn.log_softmax(logits, axis=-1)
        token_logp = jnp.take_along_axis(logp, t_c[..., None], axis=-1)[..., 0]
        return carry + (token_logp * m_c).sum(axis=-1), None

    total, _ = jax.lax.scan(one, jnp.zeros((B,), jnp.float32), (xs, ts, ms))
    return total


def preference_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta: float):
    """Sigmoid DPO loss plus the diagnostics Team 4 reports alongside it."""
    chosen_reward = beta * (policy_chosen - ref_chosen)
    rejected_reward = beta * (policy_rejected - ref_rejected)
    margin = chosen_reward - rejected_reward
    loss = -jax.nn.log_sigmoid(margin).mean()
    stats = {"margin": margin.mean(), "accuracy": (margin > 0).mean(),
             "chosen_reward": chosen_reward.mean(), "rejected_reward": rejected_reward.mean(),
             "policy_chosen_logp": policy_chosen.mean(), "policy_rejected_logp": policy_rejected.mean()}
    return loss, stats


@functools.partial(nnx.jit, static_argnames=("beta", "loss_chunk", "grad_accum"))
def train_step(model, ref, optimizer, batch, beta=0.1, loss_chunk=64, grad_accum=1):
    """Accumulate gradients over micro-batches of pairs, then take ONE Adam step.

    Reference log-probabilities are computed once for the whole batch outside the gradient,
    so the frozen model contributes no gradient and is forwarded only once per step.
    """
    pairs = batch["inputs"].shape[0] // 2
    micro = pairs // grad_accum
    ref_chosen, ref_rejected = split_sides(jax.lax.stop_gradient(
        sequence_logprob(ref, batch["inputs"], batch["targets"], batch["mask"], loss_chunk)))
    sides = {k: split_sides(v) for k, v in batch.items()}
    total_loss, total_grads, totals = jnp.float32(0), None, None
    for i in range(grad_accum):
        sl = slice(i * micro, (i + 1) * micro)
        b = {k: jnp.concatenate([c[sl], r[sl]]) for k, (c, r) in sides.items()}
        rc, rr = ref_chosen[sl], ref_rejected[sl]

        def objective(m):
            policy_chosen, policy_rejected = split_sides(
                sequence_logprob(m, b["inputs"], b["targets"], b["mask"], loss_chunk))
            loss, stats = preference_loss(policy_chosen, policy_rejected, rc, rr, beta)
            return loss / grad_accum, stats

        (loss, stats), grads = nnx.value_and_grad(objective, has_aux=True)(model)
        total_loss += loss
        total_grads = grads if total_grads is None else jax.tree.map(jnp.add, total_grads, grads)
        scaled = jax.tree.map(lambda v: v / grad_accum, stats)
        totals = scaled if totals is None else jax.tree.map(jnp.add, totals, scaled)
    grad_norm = _global_norm(total_grads)
    optimizer.update(model, total_grads)
    return total_loss, grad_norm, totals


@functools.partial(nnx.jit, static_argnames=("beta", "loss_chunk"))
def eval_step(model, ref, batch, beta=0.1, loss_chunk=64):
    policy_chosen, policy_rejected = split_sides(
        sequence_logprob(model, batch["inputs"], batch["targets"], batch["mask"], loss_chunk))
    ref_chosen, ref_rejected = split_sides(
        sequence_logprob(ref, batch["inputs"], batch["targets"], batch["mask"], loss_chunk))
    return preference_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)


def evaluate(model, ref, rows, batch_size, beta, loss_chunk):
    """Held-out preference loss and accuracy, weighted by pairs."""
    seen, loss_sum, stat_sums = 0, 0.0, {}
    for i in range(0, len(rows), batch_size):
        block = rows[i:i + batch_size]
        batch = collate_pairs(block)
        loss, stats = eval_step(model, ref, {k: jnp.asarray(v) for k, v in batch.items()},
                                beta=beta, loss_chunk=loss_chunk)
        loss_sum += float(loss) * len(block)
        for k, v in stats.items():
            stat_sums[k] = stat_sums.get(k, 0.0) + float(v) * len(block)
        seen += len(block)
    return loss_sum / seen, {k: v / seen for k, v in stat_sums.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-config", required=True)
    p.add_argument("--sft-checkpoints", required=True,
                   help="Manager root of the SFT run, ending in /checkpoints. Policy and reference both load this.")
    p.add_argument("--sft-step", type=int, required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--data", required=True,
                   help="Directory containing train.jsonl, val.jsonl, manifest.json of preference pairs")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--beta", type=float, default=0.1, help="H5 sweeps 0.1 and 0.5")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--micro-pairs", type=int, default=1, help="Preference pairs per micro-batch (2x sequences)")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-7, help="DPO runs far below the SFT rate")
    p.add_argument("--warmup", type=int, default=5, help="Smoke=5; research runs start with 100")
    p.add_argument("--loss-chunk", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pair-source", default="unspecified",
                   help="H5 condition label, e.g. sft-sampled or off-policy-<model>")
    p.add_argument("--allow-cpu", action="store_true", help="Local integration tests only")
    a = p.parse_args(argv)
    if min(a.steps, a.seq_len, a.micro_pairs, a.grad_accum, a.loss_chunk,
           a.eval_every, a.save_every) < 1 or a.lr <= 0 or not 0 <= a.warmup < a.steps:
        p.error("Invalid sizes, learning rate or warmup")
    if a.beta <= 0:
        p.error("beta must be positive")
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
        raw[split] = read_pairs(str(path))
    if pair_keys(raw["train"]) & pair_keys(raw["val"]):
        raise ValueError("Train/validation prompt overlap")
    train_rows, val_rows = (encode_pairs(raw[s], tok, a.seq_len) for s in ("train", "val"))
    base = load_config(a.base_config)
    if tok.vocab_size != base.model.vocab_size or a.seq_len > base.model.max_seq_len:
        raise ValueError("Tokenizer vocabulary/context does not match base architecture")
    sft_meta = checkpoint.read_meta(a.sft_checkpoints, a.sft_step)
    if not sft_meta.get("sft"):
        raise ValueError("DPO must start from an SFT checkpoint, not a base pretraining checkpoint")
    if sft_meta.get("template") != TEMPLATE_VERSION:
        raise ValueError("SFT checkpoint used a different chat template")
    sft_tokenizer = sft_meta.get("config", {}).get("data", {}).get("tokenizer") or base.data.tokenizer
    if Tokenizer(os.path.expandvars(sft_tokenizer)).sha256 != tok.sha256:
        raise ValueError("Tokenizer differs from the SFT checkpoint's saved tokenizer")

    run = Path(a.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=False)  # Never overwrite or implicitly resume a previous DPO run.
    batch_pairs = a.micro_pairs * a.grad_accum
    cfg = dataclasses.replace(
        base,
        model=dataclasses.replace(base.model, dtype="auto"),
        data=dataclasses.replace(base.data, tokenizer=str(Path(a.tokenizer).resolve()), seq_len=a.seq_len),
        optim=dataclasses.replace(base.optim, name="adamw", lr=a.lr, warmup_steps=a.warmup,
                                  schedule="cosine", min_lr_ratio=0.1, grad_accum=None),
        run=dataclasses.replace(base.run, name=run.name, out_dir=str(run.parent),
                                seed=a.seed, total_steps=a.steps, total_tokens=None,
                                batch_size=2 * batch_pairs, shard_params=False,
                                loss_chunk=a.loss_chunk, ckpt_every=a.save_every, tensorboard=False))
    started = time.monotonic()
    mesh = make_mesh(1)
    with jax.set_mesh(mesh):
        model = create_model(cfg, mesh)
        loaded_step = checkpoint.load_model_only(a.sft_checkpoints, model, a.sft_step)
        # The reference is a second copy of the same weights. It is never passed to an optimizer
        # and never appears in a gradient, so it stays exactly the SFT policy for the whole run.
        ref = create_model(cfg, mesh)
        ref_step = checkpoint.load_model_only(a.sft_checkpoints, ref, a.sft_step)
        if ref_step != loaded_step:
            raise AssertionError("Policy and reference loaded different steps")
        nnx.graphdef(ref)  # touch the module so a mis-constructed reference fails here, not mid-run
        tx, schedule = build_optimizer(cfg.optim, a.steps)
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        if int(optimizer.step[...]) != 0:
            raise AssertionError("DPO must start at optimizer step zero")
        cfg = dataclasses.replace(cfg, model=model.cfg)
        save_config(cfg, str(run / "config.yaml"))
        settings = vars(a) | {"sft_step_loaded": loaded_step, "optimizer_initial_step": 0,
                              "template": TEMPLATE_VERSION, "data_manifest": manifest,
                              "device": str(jax.devices()[0]), "n_params": count_params(model),
                              "train_pairs": len(train_rows), "val_pairs": len(val_rows),
                              "versions": {k: version(k) for k in ("jax", "flax", "optax", "orbax-checkpoint")},
                              "code_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                              for f in Path(__file__).parent.glob("*.py")},
                              "base_config_sha256": hashlib.sha256(Path(a.base_config).read_bytes()).hexdigest()}
        (run / "dpo_config.json").write_text(json.dumps(settings, indent=2) + "\n")

        stream = PairStream(train_rows, a.seed)
        tokens_seen = response_tokens = step = 0
        last_loss = None
        initial_val, initial_stats = evaluate(model, ref, val_rows, a.micro_pairs, a.beta, a.loss_chunk)
        # At step 0 policy and reference are identical, so every implicit reward must be exactly zero.
        if abs(initial_stats["margin"]) > 1e-4:
            raise AssertionError(f"Non-zero initial margin {initial_stats['margin']:.3e}; "
                                 "policy and reference are not the same weights")
        print(json.dumps({"step": 0, "val_loss": initial_val, **initial_stats}), flush=True)
        mgr = checkpoint.make_manager(str(run / "checkpoints"), a.save_every, keep=2, async_save=False)
        finished = False
        log = (run / "metrics.jsonl").open("w", buffering=1)
        val_loss, val_stats = initial_val, initial_stats

        def metadata():
            return {"step": step, "tokens_seen": tokens_seen, "response_tokens_seen": response_tokens,
                    "train_seconds": time.monotonic() - started, "train_loss": last_loss,
                    "val_loss": val_loss, "val_accuracy": val_stats["accuracy"],
                    "config": cfg.to_dict(), "dpo": True, "beta": a.beta,
                    "pair_source": a.pair_source, "sft_checkpoint": a.sft_checkpoints,
                    "sft_step": loaded_step, "template": TEMPLATE_VERSION}

        try:
            log.write(json.dumps({"step": 0, "val_loss": initial_val, **initial_stats}) + "\n")
            for step in range(1, a.steps + 1):
                batch = stream.take(batch_pairs)
                response_tokens += int(batch["mask"].sum())
                tokens_seen += int((batch["inputs"] != tok.pad_id).sum())
                loss, norm, stats = train_step(model, ref, optimizer,
                                               {k: jnp.asarray(v) for k, v in batch.items()},
                                               beta=a.beta, loss_chunk=a.loss_chunk, grad_accum=a.grad_accum)
                last_loss, grad_norm = float(loss), float(norm)
                if not math.isfinite(last_loss) or not math.isfinite(grad_norm):
                    raise FloatingPointError(f"Non-finite training at step {step}")
                metrics = {"step": step, "train_loss": last_loss, "grad_norm": grad_norm,
                           "lr": float(schedule(step - 1)), "tokens_seen": tokens_seen,
                           "response_tokens_seen": response_tokens,
                           "epochs": stream.examples_seen / len(train_rows),
                           **{f"train_{k}": float(v) for k, v in stats.items()},
                           "gpu_hours": (time.monotonic() - started) / 3600 if jax.default_backend() == "gpu" else 0}
                if step % a.eval_every == 0 or step == a.steps:
                    val_loss, val_stats = evaluate(model, ref, val_rows, a.micro_pairs, a.beta, a.loss_chunk)
                    metrics["val_loss"] = val_loss
                    metrics.update({f"val_{k}": v for k, v in val_stats.items()})
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
        final_val, final_stats = evaluate(model, ref, val_rows, a.micro_pairs, a.beta, a.loss_chunk)
        summary = metadata() | {
            "finished": finished, "n_params": count_params(model),
            "optimizer_final_step": int(optimizer.step[...]),
            "val_loss_before": initial_val, "val_loss": final_val,
            "val_accuracy_before": initial_stats["accuracy"], "val_accuracy": final_stats["accuracy"],
            "val_margin_before": initial_stats["margin"], "val_margin": final_stats["margin"],
            "val_loss_fell": final_val < initial_val,
            "epochs": stream.examples_seen / len(train_rows), "seed": a.seed,
            "gpu_hours": (time.monotonic() - started) / 3600 if jax.default_backend() == "gpu" else 0,
            "estimated_train_flops": 8 * count_params(model) * step * 2 * batch_pairs * a.seq_len,
            "flops_note": ("6*N policy forward+backward plus 2*N frozen reference forward, over padded "
                           "positions for both sides of every pair; excludes eval and compile"),
            "purpose": "engineering smoke test; unscanned data, not H4/H5 evidence"}
        checkpoint.write_run_summary(str(run), summary)
        print(json.dumps(summary, indent=2), flush=True)
        if not finished:
            raise SystemExit(130)


if __name__ == "__main__":
    main()
