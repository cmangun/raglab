"""Reciprocal rank fusion."""

from __future__ import annotations

from .types import Hit


def rrf(result_lists: list[list[Hit]], k: int = 60, weights: list[float] | None = None, limit: int | None = None) -> list[Hit]:
    """Fuse ranked lists by rank, not score, so differently scaled scores can be combined.

    score(chunk) = sum over lists of weight / (k + rank), rank starting at 1.
    """
    weights = weights or [1.0] * len(result_lists)
    scores: dict[str, float] = {}
    chunks = {}
    for hits, w in zip(result_lists, weights):
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk.id] = scores.get(hit.chunk.id, 0.0) + w / (k + rank)
            chunks[hit.chunk.id] = hit.chunk
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    fused = [Hit(chunks[cid], score, "fused") for cid, score in ordered]
    return fused[:limit] if limit else fused
