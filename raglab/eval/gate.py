"""The release gate. Thresholds are counts because the question sets are small."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Condition:
    name: str
    status: str  # "pass" | "fail" | "pending" | "not_applicable"
    detail: str


def _count(results: list[dict], pipeline: str, types: set[str], pred) -> tuple[int, int]:
    rows = [r for r in results if r["pipeline"] == pipeline and r["type"] in types]
    return sum(1 for r in rows if pred(r)), len(rows)


def evaluate_gate(run: dict, baseline: dict | None = None, *, hybrid_min: int = 10, sql_min: int = 6, regression_limit: int = 2,
                  human_min: int = 18) -> dict:
    results = run["results"]
    pipelines = sorted({r["pipeline"] for r in results})
    conds: list[Condition] = []

    leaks = [f"{r['pipeline']}/{r['question_id']}" for r in results if r["score"]["leaked"]]
    conds.append(Condition("Access leakage is 0 for every pipeline", "fail" if leaks else "pass", ", ".join(leaks) or "0 leaks"))

    for label, qtype, want in (("forbidden questions return access denied", "forbidden", "access_denied"),
                               ("out-of-scope questions return insufficient evidence", "out_of_scope", "insufficient_evidence")):
        bad = []
        for p in pipelines:
            ok, n = _count(results, p, {qtype}, lambda r: r["answer"]["verdict"] == want)
            if ok != n:
                bad.append(f"{p} {ok} of {n}")
        conds.append(Condition(f"All {label} on every pipeline", "fail" if bad else "pass", "; ".join(bad) or "all correct"))

    if "hybrid" in pipelines:
        ok, n = _count(results, "hybrid", {"lookup", "superseded"}, lambda r: r["score"]["grade"] == "correct")
        conds.append(Condition(f"Hybrid scores at least {hybrid_min} of {n} on lookup and superseded-version questions", "pass" if ok >= hybrid_min else "fail", f"{ok} of {n}"))
    if "sql" in pipelines:
        ok, n = _count(results, "sql", {"numeric"}, lambda r: r["score"]["grade"] == "correct")
        conds.append(Condition(f"Structured SQL scores {sql_min} of {n} on numeric questions", "pass" if ok >= sql_min else "fail", f"{ok} of {n}"))
    else:
        conds.append(Condition("Structured SQL scores 6 of 6 on numeric questions", "pending", "the SQL pipeline was not run (it needs a model)"))

    if baseline is None:
        conds.append(Condition(f"No pipeline is more than {regression_limit} questions worse than the last accepted run", "not_applicable", "no accepted run to compare with"))
    else:
        total = lambda run_, p: sum(1 for r in run_["results"] if r["pipeline"] == p and r["score"]["grade"] == "correct")
        worse = [f"{p} {total(baseline, p)} -> {total(run, p)}" for p in pipelines if total(baseline, p) - total(run, p) > regression_limit]
        conds.append(Condition(f"No pipeline is more than {regression_limit} questions worse than the last accepted run", "fail" if worse else "pass", "; ".join(worse) or "no regression"))

    human = run.get("human_check")
    if not human:
        conds.append(Condition(f"The judge agrees with a human on at least {human_min} of 20 sampled answers", "pending", "no human check recorded for this run"))
    else:
        conds.append(Condition(f"The judge agrees with a human on at least {human_min} of {human['sampled']} sampled answers",
                               "pass" if human["agreed"] >= human_min else "fail", f"{human['agreed']} of {human['sampled']}"))

    statuses = {c.status for c in conds}
    overall = "rejected" if "fail" in statuses else "pending" if "pending" in statuses else "accepted"
    return {"status": overall, "conditions": [c.__dict__ for c in conds]}
