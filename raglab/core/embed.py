"""Embedders.

`HashingEmbedder` needs no network and no model download. It is a lexical
embedder (hashed word and character n-grams), good enough to exercise every code
path and to run tests deterministically. It is not a semantic model: it cannot
match a paraphrase that shares no words. Evaluation numbers produced with it say
nothing about how a real embedding model would rank.

`OpenAICompatEmbedder` calls any OpenAI-compatible embeddings endpoint
(OpenRouter, OpenAI, Azure OpenAI, a local server).
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

import numpy as np

from .text import content_tokens


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    h = hashlib.blake2b(feature.encode(), digest_size=8).digest()
    idx = int.from_bytes(h[:4], "little") % dim
    sign = 1.0 if h[4] & 1 else -1.0
    return idx, sign


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            toks = content_tokens(text)
            feats: dict[str, int] = {}
            for t in toks:
                feats[t] = feats.get(t, 0) + 1
                for i in range(len(t) - 3):
                    g = "#" + t[i : i + 4]
                    feats[g] = feats.get(g, 0) + 1
            for a, b in zip(toks, toks[1:]):
                feats[a + "_" + b] = feats.get(a + "_" + b, 0) + 1
            for f, count in feats.items():
                idx, sign = _bucket(f, self.dim)
                out[row, idx] += sign * (1.0 + math.log(count))
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


class OpenAICompatEmbedder:
    def __init__(self, api_key: str, model: str, base_url: str = "https://openrouter.ai/api/v1", dim: int = 1536, batch: int = 64):
        import httpx

        self.name = model
        self.dim = dim
        self._model = model
        self._batch = batch
        self._client = httpx.Client(base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=60)

    def embed(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            r = self._client.post("/embeddings", json={"model": self._model, "input": texts[i : i + self._batch]})
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            rows.extend(d["embedding"] for d in data)
        arr = np.asarray(rows, dtype=np.float32)
        self.dim = arr.shape[1]
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1, norms)
