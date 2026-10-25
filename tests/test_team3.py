"""Team 3 tests: Muon optimizer, fixed-size evaluation, gradient accumulation.  Run:  python -m pytest tests -q

CPU only, 8 fake devices (same setup as test_smoke.py).
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax import nnx  # noqa: E402
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gw1b.config import DataConfig, ModelConfig, OptimConfig, RunConfig, TrainConfig, load_config  # noqa: E402
from gw1b.data import TokenDataset, write_synthetic_dataset  # noqa: E402
from gw1b.model import GW1BModel, cross_entropy  # noqa: E402
from gw1b.train import build_optimizer, evaluate, muon_labels  # noqa: E402

TINY = dict(vocab_size=512, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16, d_ff=128,
            max_seq_len=64, dtype="float32")


def _mesh():
    return jax.make_mesh((jax.device_count(),), ("data",), axis_types=(AxisType.Auto,))


def _model(cfg, mesh, seed=0):
    @nnx.jit
    def create():
        return GW1BModel(cfg, rngs=nnx.Rngs(seed), n_shards=mesh.size)
    return create()


def _batch(mesh, cfg, n=16, seed=0):
    toks = jax.device_put(jax.random.randint(jax.random.key(seed), (n, 17), 0, cfg.vocab_size),
                          NamedSharding(mesh, P("data", None)))
    return {"inputs": toks[:, :-1], "targets": toks[:, 1:]}


def _labels_by_path(model):
    leaves = jax.tree_util.tree_flatten_with_path(muon_labels(nnx.state(model, nnx.Param)))[0]
    return {jax.tree_util.keystr(p): label for p, label in leaves}


def test_muon_labels_cover_block_matrices_only():
    cfg = ModelConfig(**TINY, tie_embeddings=False)
    labels = _labels_by_path(GW1BModel(cfg, rngs=nnx.Rngs(0)))
    muon = {k for k, v in labels.items() if v == "muon"}
    adamw = {k for k, v in labels.items() if v == "adamw"}
    for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        assert sum(name in k for k in muon) == cfg.n_layers, name
    assert any("embed" in k for k in adamw) and any("lm_head" in k for k in adamw)
    assert all("norm" not in k and "embed" not in k and "lm_head" not in k for k in muon)
    assert len(muon) == 7 * cfg.n_layers


@pytest.mark.parametrize("consistent_rms", [0.2, None])
def test_muon_sharded_training_step_reduces_loss(consistent_rms):
    cfg = ModelConfig(**TINY)
    mesh = _mesh()
    with jax.set_mesh(mesh):
        model = _model(cfg, mesh)
        lr = 3e-3 if consistent_rms else 2e-2
        o = OptimConfig(name="muon", lr=lr, schedule="constant", warmup_steps=0, muon_consistent_rms=consistent_rms)
        tx, _ = build_optimizer(o, total_steps=100)
        opt = nnx.Optimizer(model, tx, wrt=nnx.Param)

        @nnx.jit
        def step(model, opt, batch):
            loss, grads = nnx.value_and_grad(lambda m: cross_entropy(m(batch["inputs"]), batch["targets"]))(model)
            opt.update(model, grads)
            return loss
        batch = _batch(mesh, cfg)
        losses = [float(step(model, opt, batch)) for _ in range(20)]
    assert np.isfinite(losses).all()
    assert losses[-1] < losses[0] - 0.5


def test_muon_train_checkpoint_resume(tmp_path):
    env = {**os.environ, "GW1B_SCRATCH": str(tmp_path), "GW1B_GROUP": str(tmp_path), "USER": "test"}
    write_synthetic_dataset(str(tmp_path / "data/synthetic/train"), n_tokens=60_000, shard_size=30_000)
    write_synthetic_dataset(str(tmp_path / "data/synthetic/val"), n_tokens=10_000, seed=1, prefix="val")
    base = [sys.executable, "-m", "gw1b.train", "--config", os.path.join(ROOT, "configs", "tiny_debug.yaml"),
            "--set", "optim.name=muon", "--set", "run.log_every=10", "--set", "run.ckpt_every=20",
            "--set", "run.eval_every=20", "--set", "run.eval_seqs=16"]
    out1 = subprocess.run(base + ["--set", "run.total_steps=20"], env=env, cwd=ROOT, capture_output=True, text=True)
    assert out1.returncode == 0, out1.stderr[-2000:]
    assert "optimizer groups: muon" in out1.stdout and "checkpoint saved at step 20" in out1.stdout
    out2 = subprocess.run(base + ["--set", "run.total_steps=30"], env=env, cwd=ROOT, capture_output=True, text=True)
    assert out2.returncode == 0, out2.stderr[-2000:]
    assert "resumed from step 20" in out2.stdout and "step 30/30" in out2.stdout
    summary = json.load(open(tmp_path / "users/test/runs/tiny_debug/summary.json"))
    assert np.isfinite(summary["val_loss"])


def test_eval_seqs_is_independent_of_micro_batch(tmp_path):
    write_synthetic_dataset(str(tmp_path / "val"), n_tokens=20_000, seed=1, prefix="val")
    ds = TokenDataset(str(tmp_path / "val"), 32)
    cfg_m = ModelConfig(**TINY)
    mesh = _mesh()
    with jax.set_mesh(mesh):
        model = _model(cfg_m, mesh)
        fixed = TrainConfig(model=cfg_m, data=DataConfig(seq_len=32), run=RunConfig(eval_seqs=64))
        vals = [evaluate(model, ds, fixed, mesh, n_batches=2, micro=m) for m in (8, 16, 32, 64)]
        legacy = TrainConfig(model=cfg_m, data=DataConfig(seq_len=32), run=RunConfig())
        old = [evaluate(model, ds, legacy, mesh, n_batches=2, micro=m) for m in (8, 32)]
    assert max(vals) - min(vals) < 1e-5            # same 64 windows whatever the micro-batch
    assert abs(old[0] - old[1]) > 1e-6             # legacy behaviour scores a different amount of data


def test_gradient_accumulation_matches_full_batch():
    cfg = ModelConfig(**TINY)
    mesh = _mesh()
    o = OptimConfig(lr=1e-3, schedule="constant", warmup_steps=0, grad_clip=0.0)
    with jax.set_mesh(mesh):
        batch = _batch(mesh, cfg, n=16)
        halves = [{k: v[i * 8:(i + 1) * 8] for k, v in batch.items()} for i in range(2)]

        def run(accum, batches):
            model = _model(cfg, mesh, seed=0)
            tx, _ = build_optimizer(o, total_steps=10, grad_accum=accum)
            opt = nnx.Optimizer(model, tx, wrt=nnx.Param)

            @nnx.jit
            def step(model, opt, b):
                grads = nnx.grad(lambda m: cross_entropy(m(b["inputs"]), b["targets"]))(model)
                opt.update(model, grads)
            for b in batches:
                step(model, opt, b)
            return model.blocks[0].mlp.up_proj.kernel[...]
        full = run(1, [batch])
        accumulated = run(2, halves)
    np.testing.assert_allclose(np.asarray(full), np.asarray(accumulated), atol=1e-6)


def test_team3_config_validation():
    cfg = load_config(None, ["optim.name=muon", "optim.muon_consistent_rms=null", "optim.muon_aux_lr_ratio=0.5",
                             "run.eval_seqs=512"])
    assert cfg.optim.name == "muon" and cfg.optim.muon_consistent_rms is None
    assert cfg.optim.muon_aux_lr_ratio == 0.5 and cfg.run.eval_seqs == 512
    with pytest.raises(AssertionError):
        load_config(None, ["optim.name=madam"])
    with pytest.raises(AssertionError):
        load_config(None, ["run.eval_seqs=0"])
