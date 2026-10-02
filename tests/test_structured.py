import pytest
from conftest import QUALITY

from raglab.core.ingest import load_tables
from raglab.core.llm import ScriptedLLM
from raglab.core.trace import TraceWriter
from raglab.core.types import Verdict
from raglab.core.verify import LexicalVerifier
from raglab.pipelines.base import run_pipeline
from raglab.pipelines.structured import SqlGuard, SqliteDatabase, SqlRejected, StructuredPipeline

TABLES = load_tables("corpus/data")


@pytest.fixture()
def db():
    return SqliteDatabase(TABLES)


@pytest.mark.parametrize("sql", [
    "DROP TABLE batch",
    "DELETE FROM batch",
    "UPDATE batch SET status = 'released'",
    "INSERT INTO batch SELECT * FROM batch",
    "SELECT * FROM batch; DELETE FROM batch",
    "SELECT * FROM sqlite_master",
    "SELECT * FROM main.batch",
    "SELECT load_extension('evil')",
    "PRAGMA table_info(batch)",
    "SELECT * FROM batch UNION SELECT * FROM batch",
    "ATTACH DATABASE 'x' AS y",
    "SELECT * FROM batch LIMIT (SELECT 5)",
    "not sql at all",
])
def test_guard_rejects(sql, db):
    with pytest.raises(SqlRejected):
        SqlGuard(db.tables()).check(sql)


def test_guard_caps_rows_and_allows_joins_and_ctes(db):
    g = SqlGuard(db.tables(), max_rows=50)
    assert g.check("SELECT * FROM batch").endswith("LIMIT 50")
    assert g.check("SELECT * FROM batch LIMIT 100000").endswith("LIMIT 50")
    assert g.check("SELECT * FROM batch LIMIT 5").endswith("LIMIT 5")
    g.check("WITH r AS (SELECT * FROM batch WHERE status = 'rejected') SELECT count(*) FROM r")
    g.check("SELECT s.name, count(*) FROM deviation d JOIN site s ON s.id = d.site_id GROUP BY s.name")


def test_database_is_read_only_even_without_the_guard(db):
    with pytest.raises(Exception, match="readonly"):
        db.execute("DELETE FROM batch")


def test_runaway_query_is_stopped():
    small = SqliteDatabase(TABLES, max_steps=20_000)
    with pytest.raises(Exception):
        small.execute("SELECT count(*) FROM batch a, batch b, batch c, deviation d")


def _pipeline(db, reply):
    return StructuredPipeline(ScriptedLLM(lambda s, u: reply), db, LexicalVerifier())


def test_answers_a_count_from_rows(db):
    expected = sum(1 for b in TABLES["batch"] if b["status"] == "hold")
    ans = run_pipeline(_pipeline(db, "SELECT count(*) FROM batch WHERE status = 'hold'"), "How many batches are on hold?", QUALITY, TraceWriter("r", "sql"))
    assert ans.verdict == Verdict.ANSWERED and str(expected) in ans.text
    assert ans.retrieved_refs == ["table:batch"] and ans.citations[0].kind == "row"


def test_no_sql_is_insufficient_evidence(db):
    ans = run_pipeline(_pipeline(db, "NO_SQL"), "What is the hold time in SOP-004?", QUALITY, TraceWriter("r", "sql"))
    assert ans.verdict == Verdict.INSUFFICIENT_EVIDENCE


def test_malicious_sql_is_an_execution_failure_not_a_decline(db):
    t = TraceWriter("r", "sql")
    ans = run_pipeline(_pipeline(db, "DELETE FROM batch"), "Remove all batches", QUALITY, t)
    assert ans.verdict == Verdict.EXECUTION_FAILED and ans.error == "SqlRejected"
    assert any(e["event_type"] == "sql_guard" and e["payload"]["allowed"] is False for e in t.events)
    assert len(db.execute("SELECT id FROM batch LIMIT 200")[1]) == 152  # nothing was deleted


# ------------------------------------------------------- column-level access

RESTRICTED = {"product.list_price_usd": frozenset({"commercial"})}
COMMERCIAL_GROUPS, QUALITY_GROUPS = frozenset({"commercial"}), frozenset({"quality"})


@pytest.mark.parametrize("sql", [
    "SELECT list_price_usd FROM product WHERE id = 'P-103'",
    "SELECT * FROM product",
    "SELECT p.* FROM product p",
    "SELECT name FROM product ORDER BY list_price_usd DESC",
    "SELECT name FROM product WHERE list_price_usd > 500",
    "SELECT max(p.list_price_usd) FROM batch b JOIN product p ON p.id = b.product_id",
    "WITH x AS (SELECT * FROM product) SELECT * FROM x",
])
def test_restricted_column_is_denied_to_other_groups(sql, db):
    from raglab.pipelines.structured import SqlAccessDenied

    g = SqlGuard(db.tables(), restricted_columns=RESTRICTED)
    with pytest.raises(SqlAccessDenied):
        g.check(sql, QUALITY_GROUPS)
    g.check(sql, COMMERCIAL_GROUPS)  # the owning group may run the same query


def test_unrestricted_queries_still_work_for_everyone(db):
    g = SqlGuard(db.tables(), restricted_columns=RESTRICTED)
    g.check("SELECT name, shelf_life_months FROM product", QUALITY_GROUPS)
    g.check("SELECT * FROM batch", QUALITY_GROUPS)
    # COUNT(*) over a table with a restricted column reads no column and must be allowed.
    g.check("SELECT COUNT(*) FROM batch b JOIN product p ON b.product_id = p.id WHERE p.name = 'Thyroid Marker Assay Kit'", QUALITY_GROUPS)
    g.check("SELECT count(*) FROM product", QUALITY_GROUPS)


def test_hidden_column_is_left_out_of_the_schema_shown_to_the_model(db):
    g = SqlGuard(db.tables(), restricted_columns=RESTRICTED)
    assert "list_price_usd" not in db.schema(g.hidden_columns(QUALITY_GROUPS))
    assert "list_price_usd" in db.schema(g.hidden_columns(COMMERCIAL_GROUPS))


def test_price_question_is_access_denied_not_answered(db):
    p = StructuredPipeline(ScriptedLLM(lambda s, u: "SELECT list_price_usd FROM product WHERE id = 'P-103'"), db, LexicalVerifier(), restricted_columns=RESTRICTED)
    ans = run_pipeline(p, "What is the list price of the Hepatic Enzyme Reagent Kit?", QUALITY, TraceWriter("r", "sql"))
    assert ans.verdict == Verdict.ACCESS_DENIED and "915" not in ans.text
    from conftest import COMMERCIAL
    ok = run_pipeline(p, "What is the list price of the Hepatic Enzyme Reagent Kit?", COMMERCIAL, TraceWriter("r", "sql"))
    assert ok.verdict == Verdict.ANSWERED
