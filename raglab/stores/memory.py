"""In-memory store: exact cosine search and BM25. For tests and small corpora."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ..core.store import FilterMode
from ..core.text import content_tokens
from ..core.types import Chunk, Hit, Principal, SearchResult


class MemoryStore:
    name = "memory"

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._emb: np.ndarray | None = None
        self._tf: list[Counter[str]] = []
        self._df: Counter[str] = Counter()
        self._len: list[int] = []

    # ------------------------------------------------------------------ write
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        assert len(chunks) == len(embeddings)
        for c in chunks:
            toks = content_tokens(c.embedding_text())
            tf = Counter(toks)
            self._tf.append(tf)
            self._len.append(len(toks))
            self._df.update(tf.keys())
        self._chunks.extend(chunks)
        self._emb = embeddings if self._emb is None else np.vstack([self._emb, embeddings])

    def count(self) -> int:
        return len(self._chunks)

    # ----------------------------------------------------------------- filter
    def _eligible(self, i: int, current_only: bool, table_rows: bool) -> bool:
        c = self._chunks[i]
        if current_only and c.status != "current":
            return False
        if not table_rows and c.is_table_row:
            return False
        return True

    # ------------------------------------------------------------------ search
    def _ranked(self, scores) -> list[int]:
        # Best first; ties break on chunk id so the order matches the Postgres store.
        return sorted(range(len(scores)), key=lambda i: (-scores[i], self._chunks[i].id))

    def vector_search(self, embedding, k, principal: Principal, *, filter_mode: FilterMode = "pre", current_only=False, table_rows=False) -> SearchResult:
        if self._emb is None:
            return SearchResult([])
        scores = self._emb @ np.asarray(embedding, dtype=np.float32)
        order = [i for i in self._ranked(scores) if self._eligible(i, current_only, table_rows)]
        if filter_mode == "pre":
            visible = [i for i in order if self._chunks[i].visible_to(principal)][:k]
            return SearchResult([Hit(self._chunks[i], float(scores[i]), "vector") for i in visible])
        top = order[:k]
        visible = [i for i in top if self._chunks[i].visible_to(principal)]
        return SearchResult([Hit(self._chunks[i], float(scores[i]), "vector") for i in visible], removed=len(top) - len(visible))

    def _bm25(self, query: str) -> np.ndarray:
        n = len(self._chunks)
        scores = np.zeros(n, dtype=np.float32)
        if n == 0:
            return scores
        avg = sum(self._len) / n
        k1, b = 1.5, 0.75
        for term in set(content_tokens(query)):
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
            for i, tf in enumerate(self._tf):
                f = tf.get(term, 0)
                if f:
                    scores[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * self._len[i] / avg))
        return scores

    def keyword_search(self, query, k, principal: Principal, *, current_only=False, table_rows=False) -> SearchResult:
        scores = self._bm25(query)
        order = [
            i
            for i in self._ranked(scores)
            if scores[i] > 0 and self._eligible(i, current_only, table_rows) and self._chunks[i].visible_to(principal)
        ][:k]
        return SearchResult([Hit(self._chunks[i], float(scores[i]), "keyword") for i in order])

    def access_profile(self, embedding, query, k, principal: Principal) -> list[bool]:
        if self._emb is None:
            return []
        vec = self._emb @ np.asarray(embedding, dtype=np.float32)
        kw = self._bm25(query)
        rows = [i for i in range(len(self._chunks)) if not self._chunks[i].is_table_row]
        by_vec = sorted(rows, key=lambda i: -vec[i])[: k * 3]
        by_kw = [i for i in sorted(rows, key=lambda i: -kw[i]) if kw[i] > 0][: k * 3]
        score: dict[int, float] = {}
        for ranked in (by_vec, by_kw):
            for rank, i in enumerate(ranked, start=1):
                score[i] = score.get(i, 0.0) + 1.0 / (60 + rank)
        top = sorted(score, key=lambda i: (-score[i], self._chunks[i].id))[:k]
        return [self._chunks[i].visible_to(principal) for i in top]

    # ------------------------------------------------------------------- read
    def get_by_ref(self, refs, principal: Principal) -> list[Chunk]:
        want = set(refs)
        return [c for c in self._chunks if c.ref in want and not c.is_table_row and c.visible_to(principal)]

    def get_document(self, document_id, principal: Principal) -> list[Chunk]:
        return [c for c in self._chunks if c.document_id == document_id and not c.is_table_row and c.visible_to(principal)]

    def all_visible(self, principal: Principal) -> list[Chunk]:
        return [c for c in self._chunks if c.visible_to(principal)]
