"""Shared 13-gram hashing. Hash is identical to the one in Yusra's playbook (Phase 3)
so hash files are interchangeable with her scanner. Pure numpy: no gw1b import."""
import re
import unicodedata
import numpy as np

N = 13
P = np.uint64(1_000_003)


def normalize(text: str) -> str:
    """PLACEHOLDER for Yusra's frozen normalization rule (D3). Keep minimal until she sends it:
    Unicode NFC + collapse whitespace runs. Change ONLY here so test and pool side stay in sync."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def ngram_hashes(tok, n: int = N) -> np.ndarray:
    t = np.asarray(tok, dtype=np.uint64)
    m = len(t) - n + 1
    if m <= 0:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(n):
            h = h * P + t[k:k + m]  # wraps mod 2**64, intended
    return h


def test_variants(tokenizer, text: str) -> list[list[int]]:
    """Tokenize with and without a leading space; the union covers start-of-document differences."""
    text = normalize(text)
    return [tokenizer.encode(text), tokenizer.encode(" " + text)]
