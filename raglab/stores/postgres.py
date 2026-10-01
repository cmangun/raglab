"""Postgres store: pgvector for vectors, tsvector for keywords, one table.

The access filter is a predicate in the same SQL statement as the search
(`groups && %(groups)s`), so restricted rows are never fetched by a "pre"
search.

Note on approximate indexes: an HNSW index is scanned first and filtered
afterwards, so a selective filter can return fewer than k rows. This store
searches exactly (no ANN index) below `ann_threshold` rows, which covers small
corpora with no recall loss. Above it, it creates an HNSW index and raises
`hnsw.ef_search`; pgvector 0.8+ iterative scans are the better fix at scale.
"""

from __future__ import annotations

import json

import numpy as np

from ..core.store import FilterMode
from ..core.text import raw_content_tokens, record_ids
from ..core.types import Chunk, Hit, Principal, SearchResult

_COLUMNS = "id, document_id, ref, text, groups, title, section, doc_type, status, is_table_row, page_no, context, metadata"


class PostgresStore:
    name = "postgres-pgvector"

    def __init__(self, dsn: str, dim: int, table: str = "rag_chunk", ann_threshold: int = 20_000):
        import psycopg
        from pgvector.psycopg import register_vector

        if not table.replace("_", "").isalnum():
            raise ValueError("table name must be alphanumeric")
        self._t = table
        self._dim = dim
        self._ann_threshold = ann_threshold
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id text PRIMARY KEY,
                document_id text NOT NULL,
                ref text NOT NULL,
                text text NOT NULL,
                groups text[] NOT NULL,
                title text NOT NULL DEFAULT '',
                section text NOT NULL DEFAULT '',
                doc_type text NOT NULL DEFAULT '',
                status text NOT NULL DEFAULT 'current',
                is_table_row boolean NOT NULL DEFAULT false,
                page_no int,
                context text NOT NULL DEFAULT '',
                metadata jsonb NOT NULL DEFAULT '{{}}',
                embedding vector({dim}) NOT NULL,
                idents text[] NOT NULL DEFAULT '{{}}',
                tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', context || ' ' || text)) STORED
            )"""
        )
        self._conn.execute(f"CREATE INDEX IF NOT EXISTS {table}_tsv ON {table} USING gin (tsv)")
        self._conn.execute(f"CREATE INDEX IF NOT EXISTS {table}_groups ON {table} USING gin (groups)")
        self._conn.execute(f"CREATE INDEX IF NOT EXISTS {table}_idents ON {table} USING gin (idents)")
        self._conn.execute(f"CREATE INDEX IF NOT EXISTS {table}_doc ON {table} (document_id)")

    # ------------------------------------------------------------------ write
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        with self._conn.cursor() as cur:
            cur.executemany(
                f"""INSERT INTO {self._t} ({_COLUMNS}, embedding, idents)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, groups = EXCLUDED.groups,
                        status = EXCLUDED.status, context = EXCLUDED.context, embedding = EXCLUDED.embedding""",
                [
                    (c.id, c.document_id, c.ref, c.text, sorted(c.groups), c.title, c.section, c.doc_type, c.status,
                     c.is_table_row, c.page_no, c.context, json.dumps(c.metadata), np.asarray(e, dtype=np.float32),
                     sorted(set(record_ids(c.embedding_text()))))
                    for c, e in zip(chunks, embeddings)
                ],
            )
        if self.count() >= self._ann_threshold:
            self._conn.execute(f"CREATE INDEX IF NOT EXISTS {self._t}_hnsw ON {self._t} USING hnsw (embedding vector_cosine_ops)")
            self._conn.execute("SET hnsw.ef_search = 200")

    def count(self) -> int:
        return self._conn.execute(f"SELECT count(*) FROM {self._t}").fetchone()[0]

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _chunk(row) -> Chunk:
        return Chunk(
            id=row[0], document_id=row[1], ref=row[2], text=row[3], groups=frozenset(row[4]), title=row[5], section=row[6],
            doc_type=row[7], status=row[8], is_table_row=row[9], page_no=row[10], context=row[11], metadata=row[12] or {},
        )

    @staticmethod
    def _scope(current_only: bool, table_rows: bool) -> str:
        sql = ""
        if current_only:
            sql += " AND status = 'current'"
        if not table_rows:
            sql += " AND NOT is_table_row"
        return sql

    # ------------------------------------------------------------------ search
    def vector_search(self, embedding, k, principal: Principal, *, filter_mode: FilterMode = "pre", current_only=False, table_rows=False) -> SearchResult:
        vec = np.asarray(embedding, dtype=np.float32)
        groups = sorted(principal.groups)
        scope = self._scope(current_only, table_rows)
        if filter_mode == "pre":
            rows = self._conn.execute(
                f"""SELECT {_COLUMNS}, 1 - (embedding <=> %s) AS score FROM {self._t}
                    WHERE groups && %s{scope} ORDER BY embedding <=> %s LIMIT %s""",
                (vec, groups, vec, k),
            ).fetchall()
            return SearchResult([Hit(self._chunk(r), float(r[-1]), "vector") for r in rows])
        # Post-filter: rank everything, keep the top k, then drop what is not visible.
        # The visibility test runs in SQL, so restricted rows still never leave the database.
        rows = self._conn.execute(
            f"""WITH top AS (
                    SELECT {_COLUMNS}, 1 - (embedding <=> %s) AS score FROM {self._t}
                    WHERE true{scope} ORDER BY embedding <=> %s LIMIT %s)
                SELECT *, (SELECT count(*) FROM top WHERE NOT (groups && %s)) AS removed
                FROM top WHERE groups && %s ORDER BY score DESC""",
            (vec, vec, k, groups, groups),
        ).fetchall()
        if rows:
            return SearchResult([Hit(self._chunk(r), float(r[-2]), "vector") for r in rows], removed=int(rows[0][-1]))
        removed = self._conn.execute(
            f"""SELECT count(*) FROM (SELECT groups FROM {self._t} WHERE true{scope} ORDER BY embedding <=> %s LIMIT %s) t
                WHERE NOT (groups && %s)""",
            (vec, k, groups),
        ).fetchone()[0]
        return SearchResult([], removed=int(removed))

    @staticmethod
    def _tsquery(query: str) -> str:
        # OR the content terms: natural-language questions rarely match every term.
        # Raw words, not our stems: Postgres applies its own English stemmer to both sides.
        terms = sorted({t for t in raw_content_tokens(query) if t.replace("-", "").replace(".", "").isalnum()})
        return " | ".join(f"'{t}'" for t in terms)

    # Postgres full-text ranking has no notion of term rarity, and its parser breaks an
    # identifier such as DEV-2025-033 into pieces. Exact record identifiers are therefore
    # indexed separately and each one a chunk shares with the query adds to its score, so a
    # query naming a record finds the passages about that record first.
    _KW_SCORE = "ts_rank_cd(tsv, q, 32) + 2 * cardinality(ARRAY(SELECT unnest(idents) INTERSECT SELECT unnest(%s::text[])))"

    def keyword_search(self, query, k, principal: Principal, *, current_only=False, table_rows=False) -> SearchResult:
        tsq, ids = self._tsquery(query), sorted(set(record_ids(query)))
        if not tsq and not ids:
            return SearchResult([])
        rows = self._conn.execute(
            f"""SELECT {_COLUMNS}, {self._KW_SCORE} AS score
                FROM {self._t}, to_tsquery('english', %s) q
                WHERE (tsv @@ q OR idents && %s::text[]) AND groups && %s{self._scope(current_only, table_rows)}
                ORDER BY score DESC, id LIMIT %s""",
            (ids, tsq or "''", ids, sorted(principal.groups), k),
        ).fetchall()
        return SearchResult([Hit(self._chunk(r), float(r[-1]), "keyword") for r in rows])

    def access_profile(self, embedding, query, k, principal: Principal) -> list[bool]:
        # Only a hashed id and a visibility flag leave the database.
        vec = np.asarray(embedding, dtype=np.float32)
        groups = sorted(principal.groups)
        by_vec = self._conn.execute(
            f"SELECT md5(id), groups && %s FROM {self._t} WHERE NOT is_table_row ORDER BY embedding <=> %s, id LIMIT %s",
            (groups, vec, k * 3),
        ).fetchall()
        tsq, ids = self._tsquery(query), sorted(set(record_ids(query)))
        by_kw = (
            self._conn.execute(
                f"""SELECT md5(id), groups && %s FROM {self._t}, to_tsquery('english', %s) q
                    WHERE (tsv @@ q OR idents && %s::text[]) AND NOT is_table_row ORDER BY {self._KW_SCORE} DESC, id LIMIT %s""",
                (groups, tsq or "''", ids, ids, k * 3),
            ).fetchall()
            if (tsq or ids)
            else []
        )
        score: dict[str, float] = {}
        visible: dict[str, bool] = {}
        for ranked in (by_vec, by_kw):
            for rank, (hid, vis) in enumerate(ranked, start=1):
                score[hid] = score.get(hid, 0.0) + 1.0 / (60 + rank)
                visible[hid] = vis
        return [visible[h] for h in sorted(score, key=lambda h: (-score[h], h))[:k]]

    # ------------------------------------------------------------------- read
    def get_by_ref(self, refs, principal: Principal) -> list[Chunk]:
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM {self._t} WHERE ref = ANY(%s) AND NOT is_table_row AND groups && %s ORDER BY id",
            (list(refs), sorted(principal.groups)),
        ).fetchall()
        return [self._chunk(r) for r in rows]

    def get_document(self, document_id, principal: Principal) -> list[Chunk]:
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM {self._t} WHERE document_id = %s AND NOT is_table_row AND groups && %s ORDER BY id",
            (document_id, sorted(principal.groups)),
        ).fetchall()
        return [self._chunk(r) for r in rows]

    def all_visible(self, principal: Principal) -> list[Chunk]:
        rows = self._conn.execute(f"SELECT {_COLUMNS} FROM {self._t} WHERE groups && %s ORDER BY id", (sorted(principal.groups),)).fetchall()
        return [self._chunk(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
