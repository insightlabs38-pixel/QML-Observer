"""qml-observer CLI.

Milestone 7 (Volume XV), Issue #50 ("CLI basic output").

Provides `inspect` and `report` subcommands operating on JSONL run logs
produced by `qml_observer.reporting.reporter.RunReporter` (Issues #48/#49):

    qml-observer inspect run.jsonl
    qml-observer report run.jsonl

Scope note: the blueprint's Volume XV also sketches `run config.yaml` and
`benchmark barren-plateau` subcommands. Implementing those would require a
declarative training-config format and a benchmark-selection harness that
no Milestone has actually specified (Milestone 7's own feature list is
scoped to JSONL logging, reports, and this CLI's *basic output* -- not a
config-driven runner). Rather than inventing that unspecified format here,
both are wired as recognized subcommands that print a clear "not yet
implemented" message and exit non-zero, so the gap is honest and
discoverable (`qml-observer run --help` still works) instead of silently
doing nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from qml_observer.reporting.export import format_compute_saved
from qml_observer.reporting.jsonl import (
    RECORD_TYPE_DIAGNOSIS,
    RECORD_TYPE_EVENT,
    RECORD_TYPE_SUMMARY,
    read_jsonl,
)

_RULE = "\u2500" * 32


def _load_records(path: str) -> list[dict[str, Any]]:
    try:
        records = list(read_jsonl(path))
    except FileNotFoundError:
        raise SystemExit(f"error: no such file: {path}") from None
    if not records:
        raise SystemExit(f"error: no records found in {path}")
    return records


def cmd_inspect(args: argparse.Namespace) -> int:
    """Print every record in a JSONL log as pretty-printed JSON."""
    records = _load_records(args.path)
    for i, record in enumerate(records):
        record_type = record.get("type", "unknown")
        print(f"--- record {i} ({record_type}) ---")
        print(json.dumps(record, indent=2, sort_keys=True))
    print(f"\n{len(records)} record(s) total.")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Print a human-readable run summary from a JSONL log (Volume XV)."""
    records = _load_records(args.path)
    events = [r for r in records if r.get("type") == RECORD_TYPE_EVENT]
    diagnoses = [r for r in records if r.get("type") == RECORD_TYPE_DIAGNOSIS]
    summaries = [r for r in records if r.get("type") == RECORD_TYPE_SUMMARY]

    if not events:
        raise SystemExit(f"error: {args.path} contains no event records")

    run_id = events[-1].get("run_id", "unknown")
    step = events[-1].get("step", len(events) - 1)
    final = summaries[-1] if summaries else (diagnoses[-1] if diagnoses else None)

    print("QML Observer")
    print(_RULE)
    print()
    print(f"Run: {run_id}")
    print(f"Step: {step:,}")
    print()

    if final is None:
        print("Status: NO DIAGNOSIS RECORDED")
    else:
        status = str(final.get("final_diagnosis") or final.get("issue") or "unknown")
        print(f"Status: {status.upper().replace('_', ' ')}")
        if final.get("degraded"):
            print("\u26a0 DIAGNOSIS DEGRADED \u2014 see logs")
        print()

        gradient = final.get("gradient")
        if gradient:
            print(f"Gradient norm:       {gradient.get('norm_l2')}")
            print(f"Gradient variance:   {gradient.get('variance')}")
        loss_curve = final.get("loss_curve_summary")
        if loss_curve:
            print(f"Loss (first -> last): {loss_curve.get('first')} -> {loss_curve.get('last')}")

        evidence = final.get("evidence") or []
        if evidence:
            print()
            print("Evidence:")
            for line in evidence:
                print(f"  - {line}")

        confidence = final.get("confidence")
        print()
        print(f"Confidence: {confidence:.0%}" if confidence is not None else "Confidence: n/a")

        recommendations = final.get("recommendations") or []
        if recommendations:
            print()
            print("Recommended next steps:")
            for rec in recommendations:
                print(f"  - {rec}")

        if "estimated_compute_saved" in final:
            saved = format_compute_saved(final["estimated_compute_saved"])
            print()
            print(f"Estimated compute saved:\n{saved}")

    print()
    print(f"Total events logged: {len(events)}")
    return 0


def cmd_not_implemented(args: argparse.Namespace) -> int:
    print(
        f"error: `qml-observer {args.command}` is not implemented yet "
        "(see Milestone 7's scope note in cli/main.py's module docstring).",
        file=sys.stderr,
    )
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qml-observer", description="QML Observer CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Print every record in a JSONL run log."
    )
    inspect_parser.add_argument("path", help="Path to a JSONL log file.")
    inspect_parser.set_defaults(func=cmd_inspect)

    report_parser = subparsers.add_parser(
        "report", help="Print a human-readable run summary from a JSONL log."
    )
    report_parser.add_argument("path", help="Path to a JSONL log file.")
    report_parser.set_defaults(func=cmd_report)

    run_parser = subparsers.add_parser(
        "run", help="(not yet implemented) Run training from a config file."
    )
    run_parser.add_argument("config", help="Path to a training config file.")
    run_parser.set_defaults(func=cmd_not_implemented)

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="(not yet implemented) Run a named benchmark."
    )
    benchmark_parser.add_argument("name", help="Benchmark name, e.g. 'barren-plateau'.")
    benchmark_parser.set_defaults(func=cmd_not_implemented)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
