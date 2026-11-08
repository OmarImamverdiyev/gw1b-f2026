import re
import unicodedata
import numpy as np

N = 13
P = np.uint64(1_000_003)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def ngram_hashes(tok, n: int = N) -> np.ndarray:
    t = np.asarray(tok, dtype=np.uint64)
    m = len(t) - n + 1
    if m <= 0:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(n):
            h = h * P + t[k:k + m]
    return h


def test_variants(tokenizer, text: str) -> list[list[int]]:
    text = normalize(text)
    return [tokenizer.encode(text), tokenizer.encode(" " + text)]
