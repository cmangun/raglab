"""Command line: ask, eval, sweep, gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .build import build_lab
from .core.trace import TraceWriter
from .core.types import Principal
from .eval.gate import evaluate_gate
from .eval.golden import access_lists, load_golden
from .eval.runner import format_matrix, hypotheses, run_eval, save_run
from .eval.sweep import leakage_sweep
from .pipelines.base import run_pipeline


def _lab(args):
    return build_lab(args.corpus, dsn=args.dsn)


def cmd_ask(args) -> int:
    lab = _lab(args)
    principal = Principal.of("cli", args.group)
    for pid in args.pipeline or list(lab.pipelines):
        trace = TraceWriter("cli", pid)
        ans = run_pipeline(lab.pipelines[pid], args.question, principal, trace)
        print(f"\n== {pid} [{ans.verdict.value}] {ans.usage.ms} ms")
        print(ans.text)
        for c in ans.citations:
            print(f"   [{c.n}] {c.ref}")
        if args.trace:
            print(json.dumps(trace.bundle()["events"], indent=1))
    return 0


def cmd_eval(args) -> int:
    lab = _lab(args)
    questions, golden_version = load_golden(Path(args.corpus) / "golden.json")
    acl = access_lists(Path(args.corpus) / "documents")
    baseline = json.loads(Path(args.baseline).read_text()) if args.baseline else None
    pipelines = [lab.pipelines[p] for p in (args.pipeline or list(lab.pipelines))]
    run = run_eval(pipelines, questions, acl, lab.judge, {**lab.record, "golden_version": golden_version}, baseline, args.run_id)
    path = save_run(run, args.out)
    print(format_matrix(run))
    print("\nhypotheses:")
    for h in hypotheses(run):
        print(f"  {h['type']:<13} expected {h['expected']:<10} best {','.join(h['best']):<28} {h['outcome']}")
    for p, why in lab.skipped.items():
        print(f"\nskipped {p}: {why}")
    print(f"\nrun written to {path}")
    return 1 if run["gate"]["status"] == "rejected" else 0


def cmd_sweep(args) -> int:
    lab = _lab(args)
    questions, _ = load_golden(Path(args.corpus) / "golden.json")
    groups = sorted({g for c in lab.chunks for g in c.groups})
    out = leakage_sweep(lab, groups, [q.text for q in questions])
    print(f"leakage sweep: {out['leaks']} leaks in {out['checks']} returned chunks checked across groups {groups}")
    for e in out["examples"]:
        print("  " + e)
    return 1 if out["leaks"] else 0


def cmd_gate(args) -> int:
    run = json.loads(Path(args.run).read_text())
    baseline = json.loads(Path(args.baseline).read_text()) if args.baseline else None
    gate = evaluate_gate(run, baseline)
    print(f"gate: {gate['status']}")
    for c in gate["conditions"]:
        print(f"  [{c['status']:^14}] {c['name']} ({c['detail']})")
    return 1 if gate["status"] == "rejected" else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="raglab")
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--dsn", default=None, help="Postgres connection string; omit to use the in-memory store")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask"); a.add_argument("question"); a.add_argument("--group", required=True)
    a.add_argument("--pipeline", action="append"); a.add_argument("--trace", action="store_true"); a.set_defaults(fn=cmd_ask)
    e = sub.add_parser("eval"); e.add_argument("--out", default="runs"); e.add_argument("--baseline"); e.add_argument("--run-id")
    e.add_argument("--pipeline", action="append"); e.set_defaults(fn=cmd_eval)
    s = sub.add_parser("sweep"); s.set_defaults(fn=cmd_sweep)
    g = sub.add_parser("gate"); g.add_argument("run"); g.add_argument("--baseline"); g.set_defaults(fn=cmd_gate)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
