"""Use case 3: questions answered from tables, not documents.

A model writes one SQL query. Nothing it writes is trusted: the guard parses the
query and rejects anything that is not a single read-only SELECT over the
allow-listed tables using allow-listed functions, then caps the row count. The
database connection is read-only as a second line of defence.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Protocol

import sqlglot
from sqlglot import exp

from ..core.llm import LLM
from ..core.trace import TraceWriter
from ..core.types import Answer, Chunk, Citation, Hit, Principal, Usage, Verdict
from ..core.verify import Verifier
from .base import MSG_DENIED, MSG_INSUFFICIENT, Components, embed_query, empty_verdict

NO_SQL = "NO_SQL"

_SAFE_FUNCTIONS = {
    "count", "sum", "avg", "min", "max", "coalesce", "round", "abs", "lower", "upper", "length", "substr", "substring",
    "cast", "date", "strftime", "julianday", "date_trunc", "extract", "nullif", "ifnull", "trim",
}
_FORBIDDEN_NODES = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command, exp.Into, exp.Merge, exp.Pragma, exp.Set, exp.Copy)


class SqlRejected(Exception):
    """The generated SQL failed the guard. Carries the reason; the SQL is never run."""


class SqlAccessDenied(SqlRejected):
    """The query is well formed but reads a column the asker may not see."""


@dataclass
class SqlGuard:
    allowed_tables: frozenset[str]
    max_rows: int = 200
    dialect: str = "sqlite"
    # "table.column" -> groups allowed to read it. Columns not listed are open to everyone.
    restricted_columns: dict[str, frozenset[str]] = field(default_factory=dict)

    def hidden_columns(self, groups: frozenset[str]) -> dict[str, set[str]]:
        """table -> columns this set of groups may not read."""
        out: dict[str, set[str]] = {}
        for key, allowed in self.restricted_columns.items():
            if not (allowed & groups):
                table, _, column = key.partition(".")
                out.setdefault(table.lower(), set()).add(column.lower())
        return out

    def check(self, sql: str, groups: frozenset[str] = frozenset()) -> str:
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
        hidden = self.hidden_columns(groups)
        used = {t.name.lower() for t in tree.find_all(exp.Table)} & set(hidden)
        if used:
            # Any star could expand to a hidden column, so it is refused on these tables.
            # COUNT(*) reads no column, so it is not a star in this sense.
            if any(not isinstance(star.parent, exp.Count) for star in tree.find_all(exp.Star)):
                raise SqlAccessDenied("SELECT * is not allowed on a table with restricted columns")
            blocked = set().union(*(hidden[t] for t in used))
            for col in tree.find_all(exp.Column):
                if col.name.lower() in blocked:
                    raise SqlAccessDenied("the query reads a restricted column")
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

    def schema(self, hidden: dict[str, set[str]] | None = None) -> str: ...

    def tables(self) -> frozenset[str]: ...

    def execute(self, sql: str) -> tuple[list[str], list[tuple]]: ...


class SqliteDatabase:
    """Loads tables into an in-memory SQLite database, then makes it read-only."""

    name = "sqlite"
    dialect = "sqlite"

    def __init__(self, tables: dict[str, list[dict]], max_steps: int = 2_000_000):
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._schema: list[tuple[str, list[tuple[str, str]]]] = []
        for name, rows in tables.items():
            cols = list(rows[0].keys())
            types = {c: ("INTEGER" if all(isinstance(r[c], int) and not isinstance(r[c], bool) for r in rows if r[c] is not None)
                         else "REAL" if all(isinstance(r[c], (int, float)) for r in rows if r[c] is not None) else "TEXT") for c in cols}
            self._conn.execute(f"CREATE TABLE {name} ({', '.join(f'{c} {types[c]}' for c in cols)})")
            self._conn.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' for _ in cols)})", [tuple(r[c] for c in cols) for r in rows])
            self._schema.append((name, [(c, types[c]) for c in cols]))
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

    def schema(self, hidden: dict[str, set[str]] | None = None) -> str:
        """Table definitions, leaving out columns the asker may not read."""
        hidden = hidden or {}
        return "\n".join(
            f"{name}({', '.join(f'{c} {t}' for c, t in cols if c.lower() not in hidden.get(name.lower(), ()))})" for name, cols in self._schema
        )

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

    PHRASE = (
        "Answer the question in one short sentence using only the query result. "
        "Use the figures exactly as given. Do not add any other number, name or explanation."
    )

    def __init__(self, llm: LLM, db: SqlDatabase, verifier: Verifier, notes: str = "", max_rows: int = 200,
                 restricted_columns: dict[str, frozenset[str]] | None = None, comp: Components | None = None, phrase: bool = False):
        self.llm, self.db, self.verifier, self.notes, self.phrase = llm, db, verifier, notes, phrase
        self.guard = SqlGuard(db.tables(), max_rows=max_rows, dialect=db.dialect, restricted_columns=restricted_columns or {})
        # With components, a question the tables cannot answer is checked against the
        # document store, so "restricted" and "not available" are told apart.
        self.comp = comp

    def _no_answer(self, question: str, principal: Principal, trace: TraceWriter, usage: Usage, refs: list[str] | None = None) -> Answer:
        if self.comp is None:
            return Answer(MSG_INSUFFICIENT, Verdict.INSUFFICIENT_EVIDENCE, retrieved_refs=refs or [], usage=usage)
        ans = empty_verdict(question, embed_query(self.comp, question), principal, self.comp, trace)
        ans.retrieved_refs, ans.usage = refs or [], usage
        return ans

    def query(self, question: str, trace: TraceWriter, usage: Usage, principal: Principal | None = None) -> tuple[str, list[str], list[tuple]] | None:
        """Generate, guard and run. Returns None when the model says the tables cannot answer."""
        groups = principal.groups if principal else frozenset()
        schema = self.db.schema(self.guard.hidden_columns(groups))
        r = self.llm.complete(self.SYSTEM.format(dialect=self.db.dialect, schema=schema, notes=self.notes), question, max_tokens=400)
        usage.add(r.usage)
        raw = r.text.strip().removeprefix("```sql").removeprefix("```").removesuffix("```").strip()
        if raw.upper().startswith(NO_SQL) or not raw:
            trace.event("sql_generate", {"model": self.llm.name, "result": "no_sql"})
            return None
        trace.event("sql_generate", {"model": self.llm.name, "sql": raw})
        try:
            safe = self.guard.check(raw, groups)
        except SqlRejected as e:
            trace.event("sql_guard", {"allowed": False, "reason": str(e)})
            raise
        trace.event("sql_guard", {"allowed": True, "sql": safe})
        columns, rows = self.db.execute(safe)
        trace.event("sql_execute", {"database": self.db.name, "columns": columns, "rows": len(rows)})
        return safe, columns, rows

    def run(self, question: str, principal: Principal, trace: TraceWriter) -> Answer:
        usage = Usage()
        try:
            out = self.query(question, trace, usage, principal)  # other rejections and database errors surface as execution_failed
        except SqlAccessDenied:
            return Answer(MSG_DENIED, Verdict.ACCESS_DENIED, usage=usage)
        if out is None:
            return self._no_answer(question, principal, trace, usage)
        safe, columns, rows = out
        tables = sorted({t.name.lower() for t in sqlglot.parse_one(safe, read=self.db.dialect).find_all(exp.Table)} & self.db.tables())
        refs = [f"table:{t}" for t in tables]
        if not rows:
            return self._no_answer(question, principal, trace, usage, refs)
        rendered = render_rows(columns, rows)
        raw = f"{rendered} [1]" if "\n" not in rendered else f"{rendered}\n[1]"
        # The verifier sees the rows as the evidence: every figure in the answer must be in them.
        evidence = Chunk(id="sql-result", document_id="sql", ref=refs[0] if refs else "table:?", text=f"{question}\n{rendered}", groups=principal.groups, title="SQL result", section=safe)
        text = raw
        if self.phrase:
            # Put the result in a sentence. If the wording adds a figure the rows do not contain, keep the raw result.
            r = self.llm.complete(self.PHRASE, f"Question: {question}\nQuery result:\n{rendered}", max_tokens=120)
            usage.add(r.usage)
            sentence = r.text.strip().replace("[1]", "").strip()
            candidate = f"{sentence.rstrip('.')}. [1]" if sentence else raw
            ok = self.verifier.verify(candidate, [Hit(evidence, 1.0, "sql")]).supported
            trace.event("phrase", {"model": self.llm.name, "kept": ok})
            text = candidate if ok else raw
        check = self.verifier.verify(text, [Hit(evidence, 1.0, "sql")])
        usage.add(check.usage)
        trace.event("verify", {"verifier": self.verifier.name, "claims": check.claims, "unsupported": len(check.unsupported), "supported": check.supported})
        return Answer(text, Verdict.ANSWERED, citations=[Citation(1, "row", r) for r in refs[:1]], retrieved_refs=refs, usage=usage, support=check.support_rate)
