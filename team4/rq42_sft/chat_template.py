"""Team 4 draft template v1. Training and generation must import this module."""

TEMPLATE_VERSION = "team4-plain-v1"


def format_prompt(prompt: str, system: str = "") -> str:
    prefix = f"### System:\n{system.strip()}\n\n" if system.strip() else ""
    return prefix + f"### Instruction:\n{prompt.strip()}\n\n### Response:\n"


def encode_example(tokenizer, prompt: str, response: str, seq_len: int,
                   system: str = "") -> dict:
    """Right-pad shifted next-token arrays; reject overflow instead of clipping answers.

    Prompt and answer are encoded separately, matching the playbook. The final
    prompt position predicts the first response token and therefore has mask=1.
    Prompt tokens remain visible to causal attention. EOS is a token, never text.
    """
    if seq_len < 2 or not prompt.strip() or not response.strip():
        raise ValueError("Empty example or invalid sequence length")
    if (tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id) != (1, 2, 3):
        raise ValueError("Expected class BOS=1, EOS=2, PAD=3")
    prompt_ids = tokenizer.encode(format_prompt(prompt, system), add_bos=True)
    response_ids = tokenizer.encode(response.strip(), add_eos=True)
    ids = prompt_ids + response_ids
    if len(ids) > seq_len + 1:
        raise OverflowError("Example exceeds context; do not silently truncate")
    target_mask = ([0] * len(prompt_ids) + [1] * len(response_ids))[1:]
    padding = seq_len - (len(ids) - 1)
    return {
        "inputs": ids[:-1] + [tokenizer.pad_id] * padding,
        "targets": ids[1:] + [tokenizer.pad_id] * padding,
        "mask": target_mask + [0] * padding,
        "input_tokens": len(ids) - 1,
        "response_tokens": len(response_ids),
    }
