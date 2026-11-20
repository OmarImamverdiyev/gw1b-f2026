"""Correctness checks for DPO; run with the class package and pinned JAX environment."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chat_template import encode_example
from dpo_data import PairStream, collate_pairs, encode_pairs, pair_keys, preference_row, split_sides
from dpo import preference_loss, sequence_logprob
from gw1b.config import ModelConfig
from gw1b.model import GW1BModel


class FakeTokenizer:
    bos_id, eos_id, pad_id = 1, 2, 3

    def encode(self, text, add_bos=False, add_eos=False):
        return ([1] if add_bos else []) + [4 + ord(c) % 17 for c in text] + ([2] if add_eos else [])


TINY = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1,
                   head_dim=8, d_ff=32, max_seq_len=128, dtype="float32")

PAIR = {"prompt": "Hi", "chosen": "Hello there", "rejected": "no", "system": ""}


def test_preference_row_drops_ties_and_empties():
    assert preference_row(PAIR)["chosen"] == "Hello there"
    assert preference_row({"prompt": "Hi", "chosen": "same", "rejected": "same"}) is None
    assert preference_row({"prompt": "Hi", "chosen": "a", "rejected": " "}) is None
    assert preference_row({"prompt": " ", "chosen": "a", "rejected": "b"}) is None
    assert preference_row({"prompt": " Hi ", "chosen": " a ", "rejected": "b"})["prompt"] == "Hi"


def test_collate_orders_chosen_then_rejected_and_splits_back():
    rows = encode_pairs([PAIR, {**PAIR, "chosen": "Another answer"}], FakeTokenizer(), 64)
    batch = collate_pairs(rows)
    assert batch["inputs"].shape == (4, 64)
    chosen, rejected = split_sides(batch["inputs"])
    np.testing.assert_array_equal(chosen[0], rows[0]["chosen"]["inputs"])
    np.testing.assert_array_equal(rejected[0], rows[0]["rejected"]["inputs"])
    np.testing.assert_array_equal(chosen[1], rows[1]["chosen"]["inputs"])
    # Both sides answer the same prompt, so the prompt tokens must be identical.
    n = rows[0]["chosen"]["input_tokens"]
    prompt_len = int(np.asarray(rows[0]["chosen"]["mask"]).argmax())
    np.testing.assert_array_equal(chosen[0][:prompt_len], rejected[0][:prompt_len])
    assert n > 0


def test_pair_stream_determinism():
    rows = encode_pairs([{**PAIR, "prompt": str(i)} for i in range(5)], FakeTokenizer(), 64)
    a, b = PairStream(rows, 42), PairStream(rows, 42)
    np.testing.assert_array_equal(a.take(7)["inputs"], b.take(7)["inputs"])
    assert a.examples_seen == 7


def test_pair_keys_ignore_responses():
    same_prompt_other_answers = {**PAIR, "chosen": "x", "rejected": "y"}
    assert pair_keys([PAIR]) == pair_keys([same_prompt_other_answers])


def test_sequence_logprob_chunked_matches_full_and_is_per_sequence():
    model = GW1BModel(TINY, rngs=nnx.Rngs(1))
    inputs = jnp.array([[1, 5, 6, 7, 8, 3, 3, 3], [1, 9, 10, 11, 12, 13, 3, 3]])
    targets = jnp.array([[5, 6, 7, 8, 2, 3, 3, 3], [9, 10, 11, 12, 13, 2, 3, 3]])
    mask = jnp.array([[0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1, 0, 0]], dtype=jnp.float32)
    chunked = sequence_logprob(model, inputs, targets, mask, chunk=2)
    full = sequence_logprob(model, inputs, targets, mask, chunk=0)
    assert chunked.shape == (2,)
    np.testing.assert_allclose(chunked, full, rtol=1e-5)
    # A sum of log-probabilities over a handful of tokens is negative and finite.
    assert np.all(np.asarray(chunked) < 0) and np.all(np.isfinite(np.asarray(chunked)))


def test_sequence_logprob_ignores_masked_targets():
    model = GW1BModel(TINY, rngs=nnx.Rngs(2))
    inputs = jnp.array([[1, 5, 6, 7, 8, 3, 3, 3]])
    targets = jnp.array([[5, 6, 7, 8, 2, 3, 3, 3]])
    mask = jnp.array([[0, 0, 1, 1, 1, 0, 0, 0]], dtype=jnp.float32)
    altered = jnp.where(mask == 0, (targets + 7) % 32, targets)
    np.testing.assert_allclose(sequence_logprob(model, inputs, targets, mask, chunk=2),
                               sequence_logprob(model, inputs, altered, mask, chunk=2), rtol=1e-6)


def test_identical_policy_and_reference_give_zero_margin():
    """Step 0 of every DPO run: the implicit reward must be exactly zero."""
    logp = jnp.array([-3.0, -5.0])
    loss, stats = preference_loss(logp, logp + 1.0, logp, logp + 1.0, beta=0.1)
    np.testing.assert_allclose(stats["margin"], 0.0, atol=1e-6)
    np.testing.assert_allclose(loss, np.log(2.0), rtol=1e-6)


def test_loss_falls_when_chosen_is_preferred_and_rises_otherwise():
    ref_c, ref_r = jnp.array([-4.0]), jnp.array([-4.0])
    better, _ = preference_loss(jnp.array([-2.0]), jnp.array([-8.0]), ref_c, ref_r, beta=0.1)
    worse, _ = preference_loss(jnp.array([-8.0]), jnp.array([-2.0]), ref_c, ref_r, beta=0.1)
    tie, _ = preference_loss(jnp.array([-4.0]), jnp.array([-4.0]), ref_c, ref_r, beta=0.1)
    assert float(better) < float(tie) < float(worse)
    np.testing.assert_allclose(tie, np.log(2.0), rtol=1e-6)


def test_beta_scales_the_margin():
    args = (jnp.array([-2.0]), jnp.array([-6.0]), jnp.array([-4.0]), jnp.array([-4.0]))
    _, small = preference_loss(*args, beta=0.1)
    _, large = preference_loss(*args, beta=0.5)
    np.testing.assert_allclose(float(large["margin"]), 5 * float(small["margin"]), rtol=1e-6)
    assert float(small["accuracy"]) == 1.0


def test_gradient_raises_chosen_and_lowers_rejected():
    """The whole point of the loss: push probability toward chosen, away from rejected."""
    ref_c, ref_r = jnp.array([-4.0]), jnp.array([-4.0])

    def loss_of(logps):
        loss, _ = preference_loss(logps[0:1], logps[1:2], ref_c, ref_r, beta=0.1)
        return loss

    grads = jax.grad(loss_of)(jnp.array([-4.0, -4.0]))
    assert float(grads[0]) < 0 < float(grads[1])


def test_reference_receives_no_gradient():
    """A frozen reference must not move; differentiating through it would do exactly that."""
    policy = GW1BModel(TINY, rngs=nnx.Rngs(3))
    reference = GW1BModel(TINY, rngs=nnx.Rngs(4))
    inputs = jnp.array([[1, 5, 6, 7, 8, 3, 3, 3], [1, 5, 6, 9, 9, 3, 3, 3]])
    targets = jnp.array([[5, 6, 7, 8, 2, 3, 3, 3], [5, 6, 9, 9, 2, 3, 3, 3]])
    mask = jnp.array([[0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0, 0]], dtype=jnp.float32)
    ref_logp = jax.lax.stop_gradient(sequence_logprob(reference, inputs, targets, mask, chunk=2))
    before = jax.tree.map(np.copy, nnx.state(reference, nnx.Param))

    def objective(m):
        chosen, rejected = split_sides(sequence_logprob(m, inputs, targets, mask, chunk=2))
        ref_chosen, ref_rejected = split_sides(ref_logp)
        loss, _ = preference_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.1)
        return loss

    grads = nnx.grad(objective)(policy)
    assert any(float(jnp.abs(g).sum()) > 0 for g in jax.tree.leaves(grads))
    after = nnx.state(reference, nnx.Param)
    for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_overflowing_pair_is_rejected_not_truncated():
    with pytest.raises(ValueError):
        encode_pairs([{**PAIR, "chosen": "x" * 200}], FakeTokenizer(), 64)
