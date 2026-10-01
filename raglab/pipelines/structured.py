"""Use case 3: questions answered from tables, not documents.

A model writes one SQL query. Nothing it writes is trusted: the guard parses the
query and rejects anything that is not a single read-only SELECT over the
allow-listed tables using allow-listed functions, then caps the row count. The
database connection is read-only as a second line of defence.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Protocol

import sqlglot
from sqlglot import exp

from ..core.llm import LLM
from ..core.trace import TraceWriter
from ..core.types import Answer, Chunk, Citation, Hit, Principal, Usage, Verdict
from ..core.verify import Verifier
from .base import MSG_INSUFFICIENT

NO_SQL = "NO_SQL"

_SAFE_FUNCTIONS = {
    "count", "sum", "avg", "min", "max", "coalesce", "round", "abs", "lower", "upper", "length", "substr", "substring",
    "cast", "date", "strftime", "julianday", "date_trunc", "extract", "nullif", "ifnull", "trim",
}
_FORBIDDEN_NODES = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command, exp.Into, exp.Merge, exp.Pragma, exp.Set, exp.Copy)


class SqlRejected(Exception):
    """The generated SQL failed the guard. Carries the reason; the SQL is never run."""


@dataclass
class SqlGuard:
    allowed_tables: frozenset[str]
    max_rows: int = 200
    dialect: str = "sqlite"

    def check(self, sql: str) -> str:
        """Return a normalised, row-limited query, or raise SqlRejected."""
        try:
            statements = [s for s in sqlglot.parse(sql.strip().rstrip(";"), read=self.dialect) if s is not None]
        except sqlglot.errors.SqlglotError as e:
            raise SqlRejected("could not be parsed") from e
        if len(statements) != 1:
            raise SqlRejected("must be exactly one statement")
        tree = statements[0]
        if not isinstance(tree, exp.Select):
            raise SqlRejected("must be a SELECT")
        for node in tree.walk():
            if isinstance(node, _FORBIDDEN_NODES):
                raise SqlRejected(f"{type(node).__name__.lower()} is not allowed")
        ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
        for table in tree.find_all(exp.Table):
            if table.db or table.catalog:
                raise SqlRejected("schema-qualified tables are not allowed")
            if table.name.lower() not in self.allowed_tables and table.name.lower() not in ctes:
                raise SqlRejected(f"table '{table.name}' is not on the allow-list")
        for fn in tree.find_all(exp.Func):
            name = (fn.name if isinstance(fn, exp.Anonymous) else fn.sql_name()).lower()
            if isinstance(fn, exp.Anonymous) and name not in _SAFE_FUNCTIONS:
                raise SqlRejected(f"function '{name}' is not on the allow-list")
        limit = tree.args.get("limit")
        current = None
        if limit is not None:
            try:
                current = int(limit.expression.name)
            except (AttributeError, ValueError):
                raise SqlRejected("LIMIT must be a literal number") from None
        if current is None or current > self.max_rows:
            tree = tree.limit(self.max_rows)
        return tree.sql(dialect=self.dialect)


class SqlDatabase(Protocol):
    name: str
    dialect: str

    def schema(self) -> str: ...

    def tables(self) -> frozenset[str]: ...

    def execute(self, sql: str) -> tuple[list[str], list[tuple]]: ...


class SqliteDatabase:
    """Loads tables into an in-memory SQLite database, then makes it read-only."""

    name = "sqlite"
    dialect = "sqlite"

    def __init__(self, tables: dict[str, list[dict]], max_steps: int = 2_000_000):
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._schema: list[str] = []
        for name, rows in tables.items():
            cols = list(rows[0].keys())
            types = {c: ("INTEGER" if all(isinstance(r[c], int) and not isinstance(r[c], bool) for r in rows if r[c] is not None)
                         else "REAL" if all(isinstance(r[c], (int, float)) for r in rows if r[c] is not None) else "TEXT") for c in cols}
            self._conn.execute(f"CREATE TABLE {name} ({', '.join(f'{c} {types[c]}' for c in cols)})")
            self._conn.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' for _ in cols)})", [tuple(r[c] for c in cols) for r in rows])
            self._schema.append(f"{name}({', '.join(f'{c} {types[c]}' for c in cols)})")
        self._names = frozenset(tables)
        self._conn.commit()
        self._conn.execute("PRAGMA query_only = ON")
        # Abort runaway queries: the handler is called every N virtual-machine steps.
        self._budget = max_steps
        self._steps = 0

        def tick() -> int:
            self._steps += 1000
            return 1 if self._steps > self._budget else 0

        self._conn.set_progress_handler(tick, 1000)

    def schema(self) -> str:
        return "\n".join(self._schema)

    def tables(self) -> frozenset[str]:
        return self._names

    def execute(self, sql: str) -> tuple[list[str], list[tuple]]:
        self._steps = 0
        cur = self._conn.execute(sql)
        return [d[0] for d in cur.description], cur.fetchall()


def render_rows(columns: list[str], rows: list[tuple], limit: int = 20) -> str:
    if len(rows) == 1 and len(columns) == 1:
        return str(rows[0][0])
    head = " | ".join(columns)
    body = "\n".join(" | ".join("" if v is None else str(v) for v in r) for r in rows[:limit])
    more = f"\n({len(rows) - limit} more rows)" if len(rows) > limit else ""
    return f"{head}\n{body}{more}"


class StructuredPipeline:
    id = "sql"
    label = "Structured data (text to SQL)"
    version = "1"

    SYSTEM = (
        "You translate a question into one {dialect} SELECT statement over the tables below. "
        "Return only the SQL, with no explanation and no code fence. "
        f"If the tables cannot answer the question, return exactly {NO_SQL}. "
        "Dates are ISO strings (YYYY-MM-DD). Never modify data.\n\nTables:\n{schema}\n\n{notes}"
    )

    def __init__(self, llm: LLM, db: SqlDatabase, verifier: Verifier, notes: str = "", max_rows: int = 200):
        self.llm, self.db, self.verifier, self.notes = llm, db, verifier, notes
        self.guard = SqlGuard(db.tables(), max_rows=max_rows, dialect=db.dialect)

    def query(self, question: str, trace: TraceWriter, usage: Usage) -> tuple[str, list[str], list[tuple]] | None:
        """Generate, guard and run. Returns None when the model says the tables cannot answer."""
        r = self.llm.complete(self.SYSTEM.format(dialect=self.db.dialect, schema=self.db.schema(), notes=self.notes), question, max_tokens=400)
        usage.add(r.usage)
        raw = r.text.strip().removeprefix("```sql").removeprefix("```").removesuffix("```").strip()
        if raw.upper().startswith(NO_SQL) or not raw:
            trace.event("sql_generate", {"model": self.llm.name, "result": "no_sql"})
            return None
        trace.event("sql_generate", {"model": self.llm.name, "sql": raw})
        try:
            safe = self.guard.check(raw)
        except SqlRejected as e:
            trace.event("sql_guard", {"allowed": False, "reason": str(e)})
            raise
        trace.event("sql_guard", {"allowed": True, "sql": safe})
        columns, rows = self.db.execute(safe)
        trace.event("sql_execute", {"database": self.db.name, "columns": columns, "rows": len(rows)})
        return safe, columns, rows

    def run(self, question: str, principal: Principal, trace: TraceWriter) -> Answer:
        usage = Usage()
        out = self.query(question, trace, usage)  # SqlRejected and database errors surface as execution_failed
        if out is None:
            return Answer(MSG_INSUFFICIENT, Verdict.INSUFFICIENT_EVIDENCE, usage=usage)
        safe, columns, rows = out
        tables = sorted({t.name.lower() for t in sqlglot.parse_one(safe, read=self.db.dialect).find_all(exp.Table)} & self.db.tables())
        refs = [f"table:{t}" for t in tables]
        if not rows:
            return Answer(MSG_INSUFFICIENT, Verdict.INSUFFICIENT_EVIDENCE, retrieved_refs=refs, usage=usage)
        rendered = render_rows(columns, rows)
        text = f"{rendered} [1]" if "\n" not in rendered else f"{rendered}\n[1]"
        # The verifier sees the rows as the evidence: every figure in the answer must be in them.
        evidence = Chunk(id="sql-result", document_id="sql", ref=refs[0] if refs else "table:?", text=rendered, groups=principal.groups, title="SQL result", section=safe)
        check = self.verifier.verify(text, [Hit(evidence, 1.0, "sql")])
        usage.add(check.usage)
        trace.event("verify", {"verifier": self.verifier.name, "claims": check.claims, "unsupported": len(check.unsupported), "supported": check.supported})
        return Answer(text, Verdict.ANSWERED, citations=[Citation(1, "row", r) for r in refs[:1]], retrieved_refs=refs, usage=usage, support=check.support_rate)
