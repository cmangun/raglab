"""The chunk store contract.

Two rules are built into the signature rather than left to callers:

1. Every search takes a `Principal`. There is no way to search without one.
2. A store never returns a chunk the principal cannot see. With
   `filter_mode="pre"` the access filter is part of the search itself. With
   `filter_mode="post"` the store searches everything, then drops what the
   principal cannot see and reports only how many it dropped. Post-filtering is
   the naive approach; it exists so its cost (lost recall) can be measured.
"""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np

from .types import Chunk, Principal, SearchResult

FilterMode = Literal["pre", "post"]


class ChunkStore(Protocol):
    name: str

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None: ...

    def vector_search(
        self,
        embedding: np.ndarray,
        k: int,
        principal: Principal,
        *,
        filter_mode: FilterMode = "pre",
        current_only: bool = False,
        table_rows: bool = False,
    ) -> SearchResult: ...

    def keyword_search(
        self,
        query: str,
        k: int,
        principal: Principal,
        *,
        current_only: bool = False,
        table_rows: bool = False,
    ) -> SearchResult: ...

    def get_by_ref(self, refs: list[str], principal: Principal) -> list[Chunk]: ...

    def get_document(self, document_id: str, principal: Principal) -> list[Chunk]: ...

    def access_profile(self, embedding: np.ndarray, query: str, k: int, principal: Principal) -> list[bool]:
        """Visibility of the top-k candidates, ignoring access, in rank order.

        True means the principal may see that candidate. Used only to tell
        `access_denied` from `insufficient_evidence`. It carries no ids, text or
        scores: nothing about a restricted chunk except that one ranked there.
        """
        ...

    def count(self) -> int: ...
