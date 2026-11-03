"""Correctness checks; run with the class package and pinned JAX environment."""
import dataclasses
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from chat_template import TEMPLATE_VERSION, encode_example, format_prompt
from sft_data import BatchStream, collate, prompt_key, single_turn
from sft import main, train_step
from gw1b import checkpoint
from gw1b.config import ModelConfig, TrainConfig, DataConfig, save_config
from gw1b.model import GW1BModel, cross_entropy, chunked_cross_entropy


class FakeTokenizer:
    bos_id, eos_id, pad_id = 1, 2, 3

    def encode(self, text, add_bos=False, add_eos=False):
        return ([1] if add_bos else []) + [4 + ord(c) % 17 for c in text] + ([2] if add_eos else [])


TINY = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1,
                   head_dim=8, d_ff=32, max_seq_len=128, dtype="float32")


def test_shift_eos_padding_and_prompt_boundary():
    tok = FakeTokenizer()
    prompt_ids = tok.encode(format_prompt("Hi"), add_bos=True)
    response_ids = tok.encode("OK", add_eos=True)
    e = encode_example(tok, "Hi", "OK", 64)
    n = len(prompt_ids)
    assert e["mask"][:n - 1] == [0] * (n - 1)
    assert e["targets"][n - 1] == response_ids[0]
    assert e["mask"][n - 1:n - 1 + len(response_ids)] == [1] * len(response_ids)
    assert e["targets"][e["input_tokens"] - 1] == tok.eos_id
    assert sum(e["mask"]) == len(response_ids)
    assert e["mask"][e["input_tokens"]:] == [0] * (64 - e["input_tokens"])
    assert all(len(e[k]) == 64 for k in ("inputs", "targets", "mask"))


def test_overflow_rejected_and_no_empty_answer():
    with pytest.raises(OverflowError):
        encode_example(FakeTokenizer(), "Hi", "x" * 100, 64)
    with pytest.raises(ValueError):
        encode_example(FakeTokenizer(), "Hi", " ", 64)


def test_single_turn_preserves_system_and_uses_first_exchange():
    row = {"messages": [{"role": role, "content": text} for role, text in
                        [("system", "Be brief"), ("user", "Hi"), ("assistant", "Hello"),
                         ("user", "Continue"), ("assistant", "Goodbye")]]}
    assert single_turn(row) == {"system": "Be brief", "prompt": "Hi", "response": "Hello"}
    assert "Be brief" in format_prompt("Hi", "Be brief")
    assert single_turn({"messages": [{"role": "assistant", "content": "Hi"}]}) is None


def test_prompt_hash_normalizes_and_ignores_answer():
    assert prompt_key("  HELLO\n world ") == prompt_key("hello world")
    assert prompt_key("hello", "A") != prompt_key("hello", "B")


def test_batch_stream_determinism_and_epoch_tail():
    rows = [encode_example(FakeTokenizer(), str(i), str(i), 64) for i in range(5)]
    a = BatchStream(rows, 42)
    for _ in range(3):
        for key, value in a.take(3).items():
            # A single take is compared below, since all arrays share the same shuffle.
            assert value.shape == (3, 64)
        assert a.examples_seen % 3 == 0
    c, d = BatchStream(rows, 42), BatchStream(rows, 42)
    np.testing.assert_array_equal(c.take(11)["inputs"], d.take(11)["inputs"])
    assert c.examples_seen == 11


def test_masked_target_loss_and_gradient_zero():
    logits = jax.random.normal(jax.random.key(1), (1, 8, 32))
    targets = jnp.arange(8)[None, :]
    mask = jnp.array([[0, 0, 0, 1, 1, 0, 0, 0]], dtype=jnp.float32)
    altered = jnp.where(mask == 0, (targets + 7) % 32, targets)
    np.testing.assert_allclose(cross_entropy(logits, targets, mask), cross_entropy(logits, altered, mask))
    grads = jax.grad(cross_entropy)(logits, targets, mask)
    np.testing.assert_array_equal(np.asarray(grads)[np.asarray(mask) == 0], 0)
    assert float(jnp.linalg.norm(grads[:, 3:5])) > 0


def test_chunked_response_loss_matches_full():
    h = jax.random.normal(jax.random.key(2), (2, 8, 16))
    w = jax.random.normal(jax.random.key(3), (32, 16))
    targets = jnp.arange(16).reshape(2, 8)
    mask = jnp.array([[0, 0, 1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1]])
    np.testing.assert_allclose(chunked_cross_entropy(h, w, targets, mask, 2),
                               chunked_cross_entropy(h, w, targets, mask, 0), rtol=1e-6)


def test_causal_padding_invariance_without_hiding_prompt():
    model = GW1BModel(TINY, rngs=nnx.Rngs(1))
    inputs = jnp.array([[1, 5, 6, 7, 8, 3, 3, 3]])
    targets = jnp.array([[5, 6, 7, 8, 2, 3, 3, 3]])
    mask = jnp.array([[0, 0, 1, 1, 1, 0, 0, 0]])
    loss = model.loss(inputs, targets, mask, chunk=2)
    changed_padding = inputs.at[:, 5:].set(20)
    np.testing.assert_allclose(loss, model.loss(changed_padding, targets, mask, chunk=2), atol=1e-6)
    changed_prompt = inputs.at[:, 1].set(18)
    assert abs(float(loss - model.loss(changed_prompt, targets, mask, chunk=2))) > 1e-7


def test_token_weighted_accumulation_matches_full_batch():
    models = [GW1BModel(TINY, rngs=nnx.Rngs(5)) for _ in range(2)]
    optimizers = [nnx.Optimizer(m, optax.sgd(0.01), wrt=nnx.Param) for m in models]
    batch = {"inputs": jnp.array([[1, 4, 5, 6, 3, 3, 3, 3], [1, 8, 9, 10, 11, 12, 13, 14]]),
             "targets": jnp.array([[4, 5, 6, 2, 3, 3, 3, 3], [8, 9, 10, 11, 12, 13, 14, 2]]),
             "mask": jnp.array([[0, 0, 1, 1, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1]], dtype=jnp.float32)}
    results = [train_step(m, o, batch, loss_chunk=2, grad_accum=accum)
               for m, o, accum in zip(models, optimizers, [1, 2])]
    np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-6)
    for left, right in zip(jax.tree.leaves(nnx.state(models[0], nnx.Param)),
                           jax.tree.leaves(nnx.state(models[1], nnx.Param))):
        np.testing.assert_allclose(left, right, atol=1e-7, rtol=1e-5)
    assert all(int(o.step[...]) == 1 for o in optimizers)


def test_end_to_end_fresh_optimizer_and_checkpoint(tmp_path):
    tokpath = os.environ.get("SFT_TEST_TOKENIZER")
    if not tokpath:
        pytest.skip("Set SFT_TEST_TOKENIZER to the standard class tokenizer")
    from gw1b.tokenizer import Tokenizer
    tok = Tokenizer(tokpath)
    cfg = TrainConfig(model=dataclasses.replace(TINY, vocab_size=tok.vocab_size),
                      data=DataConfig(tokenizer=tokpath, seq_len=64))
    base = tmp_path / "base"
    base.mkdir()
    save_config(cfg, str(base / "config.yaml"))
    model = GW1BModel(cfg.model, rngs=nnx.Rngs(42))
    mgr = checkpoint.make_manager(str(base / "checkpoints"), 1, async_save=False)
    checkpoint.save(mgr, 23, model, None, {"step": 23, "config": cfg.to_dict()}, force=True)
    mgr.close()
    data = tmp_path / "data"
    data.mkdir()
    rows = [{"prompt": f"Reply with the word yes. Example {i}.", "response": "yes"} for i in range(6)]
    files = {}
    for split, content in [("train", rows[:4]), ("val", rows[4:])]:
        path = data / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in content))
        files[split] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (data / "manifest.json").write_text(json.dumps({"template": TEMPLATE_VERSION, "seq_len": 64,
                         "tokenizer_sha256prefix": tok.sha256, "files": files}))
    run = tmp_path / "sft"
    args = ["--base-config", str(base / "config.yaml"), "--base-checkpoints", str(base / "checkpoints"),
            "--base-step", "23", "--tokenizer", tokpath, "--data", str(data), "--run-dir", str(run),
            "--steps", "3", "--seq-len", "64", "--micro-batch", "1", "--grad-accum", "2",
            "--warmup", "1", "--lr", "0.001", "--loss-chunk", "16", "--eval-every", "3", "--allow-cpu"]
    main(args)
    summary = json.loads((run / "summary.json").read_text())
    settings = json.loads((run / "sft_config.json").read_text())
    assert summary["finished"] and summary["step"] == 3 and summary["optimizer_final_step"] == 3
    assert settings["optimizer_initial_step"] == 0 and settings["base_step_loaded"] == 23
    assert summary["train_probe_loss_after"] < summary["train_probe_loss_before"]
    restored = GW1BModel(cfg.model, rngs=nnx.Rngs(99))
    assert checkpoint.load_model_only(str(run / "checkpoints"), restored) == 3
    assert checkpoint.latest_step(str(base / "checkpoints")) == 23
    with pytest.raises(FileExistsError):
        main(args)
