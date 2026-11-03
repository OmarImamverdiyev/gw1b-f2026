"""Generate benign development examples with the identical SFT prompt template."""
import argparse
import json
import os
from pathlib import Path

from chat_template import format_prompt, TEMPLATE_VERSION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prompt", action="append", required=True)
    p.add_argument("--max-new-tokens", type=int, default=64)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        p.error("Submit sampling through gw1b run on a compute node")
    from gw1b.evaluate import load_run
    from gw1b.generate import generate
    from gw1b.tokenizer import Tokenizer
    model, cfg, step = load_run(a.run)
    tok = Tokenizer(cfg.data.tokenizer)
    texts = generate(model, tok, [format_prompt(prompt) for prompt in a.prompt],
                     max_new_tokens=a.max_new_tokens, temperature=0, add_bos=True)
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        for prompt, text in zip(a.prompt, texts):
            f.write(json.dumps({"prompt": prompt, "response": text, "step": step,
                                "template": TEMPLATE_VERSION, "eos_id": tok.eos_id,
                                "max_new_tokens": a.max_new_tokens}) + "\n")


if __name__ == "__main__":
    main()
