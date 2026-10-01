"""The access contract, run against both stores."""

from conftest import COMMERCIAL, QUALITY

from raglab.eval.sweep import leakage_sweep


def _vec(lab, text):
    return lab.comp.embedder.embed([text])[0]


def test_pre_filter_never_returns_restricted(any_lab):
    q = "list price per kit for the Hepatic Enzyme Reagent Kit"
    for mode in ("pre", "post"):
        res = any_lab.comp.store.vector_search(_vec(any_lab, q), 10, QUALITY, filter_mode=mode)
        assert all(h.chunk.visible_to(QUALITY) for h in res.hits)
    assert all(h.chunk.visible_to(QUALITY) for h in any_lab.comp.store.keyword_search(q, 10, QUALITY).hits)


def test_post_filter_reports_a_count_and_loses_results(any_lab):
    q = "Commercial memo price list restricted to the Commercial group"
    store = any_lab.comp.store
    post = store.vector_search(_vec(any_lab, q), 8, QUALITY, filter_mode="post")
    pre = store.vector_search(_vec(any_lab, q), 8, QUALITY, filter_mode="pre")
    assert post.removed > 0
    assert len(post.hits) == 8 - post.removed  # the naive approach returns fewer than k
    assert len(pre.hits) == 8 and pre.removed == 0  # filtering inside the query still fills k


def test_owner_sees_their_own_documents(any_lab):
    q = "maximum discount on distributor contracts"
    hits = any_lab.comp.store.keyword_search(q, 5, COMMERCIAL).hits
    assert any(h.chunk.document_id == "MEMO-05" for h in hits)


def test_direct_fetch_respects_access(any_lab):
    store = any_lab.comp.store
    assert store.get_document("MEMO-01", QUALITY) == []
    assert store.get_document("MEMO-01", COMMERCIAL)
    assert store.get_by_ref(["AF-03#finding"], COMMERCIAL) == []
    assert store.get_by_ref(["AF-03#finding"], QUALITY)


def test_current_only_excludes_superseded(any_lab):
    q = "SOP-008 cumulative excursion limit above 8 °C"
    store = any_lab.comp.store
    with_old = {h.chunk.document_id for h in store.keyword_search(q, 20, QUALITY).hits}
    current = {h.chunk.document_id for h in store.keyword_search(q, 20, QUALITY, current_only=True).hits}
    assert "SOP-008-v1" in with_old and "SOP-008-v1" not in current and "SOP-008-v2" in current


def test_table_rows_only_when_asked(any_lab):
    q = "Shelf life (months) Respiratory Panel Master Mix"
    store = any_lab.comp.store
    assert not any(h.chunk.is_table_row for h in store.keyword_search(q, 20, QUALITY).hits)
    assert any(h.chunk.is_table_row for h in store.keyword_search(q, 20, QUALITY, table_rows=True).hits)


def test_access_profile_carries_only_booleans(any_lab):
    q = "What is the maximum discount allowed on distributor contracts?"
    profile = any_lab.comp.store.access_profile(_vec(any_lab, q), q, 8, QUALITY)
    assert profile and all(isinstance(v, bool) for v in profile)
    assert profile[0] is False  # the best match is restricted, and that is all the caller learns
    assert all(any_lab.comp.store.access_profile(_vec(any_lab, q), q, 8, COMMERCIAL)[:1])


def test_both_stores_rank_vectors_identically(lab, pg_lab):
    for q in ["retained samples per batch", "freezer alarm test interval", "impact assessment due"]:
        a = [h.chunk.id for h in lab.comp.store.vector_search(_vec(lab, q), 8, QUALITY).hits]
        b = [h.chunk.id for h in pg_lab.comp.store.vector_search(_vec(pg_lab, q), 8, QUALITY).hits]
        assert a == b


def test_leakage_sweep_finds_nothing(any_lab):
    out = leakage_sweep(any_lab, ["commercial", "quality"])
    assert out["checks"] > 2000 and out["leaks"] == 0, out["examples"]
