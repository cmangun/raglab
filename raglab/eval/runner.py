"""Run every pipeline over the golden set and write one self-describing run file."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from ..core.trace import TraceWriter, verify_chain
from ..pipelines.base import Pipeline, run_pipeline
from .gate import evaluate_gate
from .golden import Question
from .metrics import Judge, score_answer


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "uncommitted"


def run_eval(pipelines: list[Pipeline], questions: list[Question], acl: dict, judge: Judge, record: dict, baseline: dict | None = None,
             run_id: str | None = None) -> dict:
    run_id = run_id or datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    results = []
    for p in pipelines:
        for q in questions:
            trace = TraceWriter(run_id, p.id, q.id)
            answer = run_pipeline(p, q.text, q.principal, trace)
            bundle = trace.bundle()
            ok, _ = verify_chain(bundle)
            score = score_answer(q, answer, bundle, acl, judge)
            results.append({"pipeline": p.id, "question_id": q.id, "type": q.type, "group": q.group, "question": q.text,
                            "expected_verdict": q.expected_verdict.value, "hypothesis": q.hypothesis,
                            "answer": answer.to_dict(), "score": score.to_dict(), "trace": bundle, "trace_chain_ok": ok})
    run = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Everything needed to repeat or explain the run.
        "record": {**record, "code_version": _git_sha(), "python": platform.python_version(), "judge": judge.name,
                   "pipelines": {p.id: p.version for p in pipelines}},
        "results": results,
    }
    run["summary"] = summarise(run)
    run["gate"] = evaluate_gate(run, baseline)
    return run


def summarise(run: dict) -> dict:
    out: dict = {}
    for r in run["results"]:
        p = out.setdefault(r["pipeline"], {"by_type": {}, "leaks": 0, "verifier_interventions": 0, "execution_failures": 0, "_ms": [], "cost_usd": 0.0, "answers": 0})
        t = p["by_type"].setdefault(r["type"], {"questions": 0, "correct": 0, "partial": 0, "verdict_ok": 0, "_recall": []})
        s = r["score"]
        t["questions"] += 1
        t["correct"] += s["grade"] == "correct"
        t["partial"] += s["grade"] == "partial"
        t["verdict_ok"] += s["verdict_ok"]
        if s["recall"] is not None:
            t["_recall"].append(s["recall"])
        p["answers"] += 1
        p["leaks"] += s["leaked"]
        p["verifier_interventions"] += s["verifier_changed"]
        p["execution_failures"] += s["execution_failed"]
        p["_ms"].append(r["answer"]["usage"]["ms"])
        p["cost_usd"] += r["answer"]["usage"]["cost_usd"]
    for p in out.values():
        ms = sorted(p.pop("_ms"))
        p["latency_ms_median"] = round(median(ms), 1)
        p["latency_ms_p95"] = round(ms[min(len(ms) - 1, int(len(ms) * 0.95))], 1)
        p["cost_usd"] = round(p["cost_usd"], 4)
        for t in p["by_type"].values():
            rec = t.pop("_recall")
            t["recall"] = round(sum(rec) / len(rec), 2) if rec else None
    return out


def hypotheses(run: dict) -> list[dict]:
    """For each question type, which pipeline was expected to do best and which did. Annotation only."""
    out = []
    types: dict[str, str] = {}
    for r in run["results"]:
        types.setdefault(r["type"], r["hypothesis"])
    for qtype, expected in types.items():
        if expected in ("all", ""):
            continue
        scores = {p: s["by_type"][qtype]["correct"] for p, s in run["summary"].items() if qtype in s["by_type"]}
        best = max(scores.values()) if scores else 0
        winners = sorted(p for p, v in scores.items() if v == best)
        out.append({"type": qtype, "expected": expected, "best": winners, "scores": scores,
                    "outcome": "not run" if expected not in scores else "not confirmed" if (best == 0 or expected not in winners)
                    else "confirmed" if winners == [expected] else "tied"})
    return out


def format_matrix(run: dict) -> str:
    types = ["lookup", "superseded", "multi_hop", "numeric", "relationship", "table", "out_of_scope", "forbidden"]
    lines = ["pipeline      " + "".join(f"{t[:11]:>13}" for t in types) + "   leaks  verif  fail   ms(med)"]
    for p, s in run["summary"].items():
        cells = []
        for t in types:
            b = s["by_type"].get(t)
            if not b:
                cells.append(f"{'-':>13}")
            elif t in ("out_of_scope", "forbidden"):
                cells.append(f"{str(b['verdict_ok']) + '/' + str(b['questions']):>13}")
            else:
                cells.append(f"{str(b['correct']) + '/' + str(b['questions']) + ' r' + (format(b['recall'], '.2f') if b['recall'] is not None else '-'):>13}")
        lines.append(f"{p:<14}" + "".join(cells) + f"{s['leaks']:>8}{s['verifier_interventions']:>7}{s['execution_failures']:>6}{s['latency_ms_median']:>10}")
    lines.append("")
    lines.append(f"gate: {run['gate']['status']}")
    for c in run["gate"]["conditions"]:
        lines.append(f"  [{c['status']:^14}] {c['name']} ({c['detail']})")
    return "\n".join(lines)


def save_run(run: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{run['run_id']}.json"
    path.write_text(json.dumps(run, indent=1, ensure_ascii=False), encoding="utf-8")
    return path
