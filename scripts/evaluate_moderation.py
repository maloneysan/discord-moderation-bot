#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from discord_moderation_bot.engine import ModerationEngine  # noqa: E402
from discord_moderation_bot.evaluation import (  # noqa: E402
    evaluate_cases,
    load_evaluation_cases,
)
from discord_moderation_bot.service import (  # noqa: E402
    GroqModerationService,
    LocalModerationService,
)


async def _run(args: argparse.Namespace) -> int:
    rules_path = PROJECT_ROOT / "config" / "rules.json"
    dataset_path = Path(args.dataset)
    engine = ModerationEngine.from_json(rules_path)
    cases = load_evaluation_cases(dataset_path)
    if args.case_id:
        selected = frozenset(args.case_id)
        cases = tuple(case for case in cases if case.case_id in selected)
        missing = selected - {case.case_id for case in cases}
        if missing:
            print("unknown case ids=" + ",".join(sorted(missing)), file=sys.stderr)
            return 2
    if args.limit is not None:
        cases = cases[: args.limit]

    if args.backend == "groq":
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            print("GROQ_API_KEY is required for --backend groq", file=sys.stderr)
            return 2
        service = GroqModerationService(api_key, engine)
    else:
        service = LocalModerationService(engine)

    try:
        report = await evaluate_cases(service, cases)
    finally:
        await service.close()

    print(f"cases={report.case_count}")
    print(f"exact_accuracy={report.exact_accuracy:.3f}")
    print(f"precision={report.precision:.3f}")
    print(f"recall={report.recall:.3f}")
    print(f"f1={report.f1:.3f}")
    if report.mismatched_case_ids:
        print("mismatches=" + ",".join(report.mismatched_case_ids))
    return int(
        report.precision < args.fail_under_precision
        or report.recall < args.fail_under_recall
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="本文を出力せず、人工例文でモデレーション精度を評価します。"
    )
    parser.add_argument("--backend", choices=("local", "groq"), default="local")
    parser.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "config" / "evaluation_cases.json"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--fail-under-precision", type=float, default=0.0)
    parser.add_argument("--fail-under-recall", type=float, default=0.0)
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
