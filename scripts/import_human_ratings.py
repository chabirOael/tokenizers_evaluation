#!/usr/bin/env python3
"""Bring a filled blind CSV (from scripts/build_human_eval_v2.py) into the Rate tab's ratings files.

    .venv/bin/python scripts/import_human_ratings.py --experiment outputs/experiments/qwen_native_vs_araroopat \
        --set human_v2 --rater Mariam --csv /path/to/human_v2_qwen_blind_Mariam.csv

Each row with a non-empty ``score`` goes through the same ``submit`` the page uses (score must be 1–5; ``flags`` is a
``|``-separated subset of repetition, wrong_language, empty, off_topic, truncated; ``note`` up to 500 characters), so a
rating imported here is indistinguishable from one entered in the browser. Rows without a score are reported, not written.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arabic_eval.tools.freeform_rating import FreeformError, submit  # noqa: E402


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def read_rows(path: Path):
    """Rows as dicts keyed by the header names, from the blind CSV or from the Excel workbook (sheet ``Ratings``)."""
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        from openpyxl import load_workbook
        wb = load_workbook(path, data_only=True, read_only=True)
        ws = wb["Ratings"] if "Ratings" in wb.sheetnames else wb[wb.sheetnames[0]]
        rows = ws.iter_rows(values_only=True)
        header = None
        for r in rows:
            if r and "item_id" in r and "score" in r:
                header = [_cell(h) for h in r]
                break
        if header is None:
            sys.exit(f"{path}: no header row with item_id and score on sheet {ws.title!r}")
        for r in rows:
            d = {h: _cell(v) for h, v in zip(header, r)}
            if d.get("item_id"):
                yield d
    else:
        with path.open(newline="", encoding="utf-8-sig") as f:
            yield from csv.DictReader(f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--experiment", required=True, help="e.g. outputs/experiments/qwen_native_vs_araroopat")
    ap.add_argument("--set", default="human_v2")
    ap.add_argument("--rater", required=True, help="the annotator's name, used as the ratings file name")
    ap.add_argument("--csv", required=True, type=Path, help="the returned file: the blind CSV or the Excel workbook (.xlsx)")
    args = ap.parse_args()

    ok, skipped, errors = 0, [], []
    for row in read_rows(args.csv):
            score = (row.get("score") or "").strip()
            if not score:
                skipped.append(row["item_id"])
                continue
            flags = [x.strip() for x in (row.get("flags") or "").replace(",", "|").split("|") if x.strip()]
            try:
                submit(ROOT, args.experiment, args.set, args.rater, row["item_id"], score, flags, row.get("note") or "")
                ok += 1
            except FreeformError as e:
                errors.append(f"{row['item_id']}: {e}")
    print(f"imported {ok} ratings for {args.rater!r} into {args.experiment}/freeform_rating/{args.set}.ratings/")
    if skipped:
        print(f"{len(skipped)} rows without a score were not written: {', '.join(skipped[:10])}{' …' if len(skipped) > 10 else ''}")
    if errors:
        print("rows refused:\n  " + "\n  ".join(errors))
        sys.exit(1)


if __name__ == "__main__":
    main()
