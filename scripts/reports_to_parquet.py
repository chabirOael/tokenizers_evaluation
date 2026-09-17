#!/usr/bin/env python
"""Convert the CSV reports of older runs to Parquet (and back, for a spreadsheet).

Failure reports and UNK reports used to be written as UTF-8-BOM CSV. They are
Parquet now: a fraction of the bytes, one ``pd.read_parquet`` to load, and
column projection so a reader can filter without decoding the Arabic text.
This script migrates what is already on disk so old runs load with the same
tooling as new ones.

    # see what would happen
    .venv/bin/python scripts/reports_to_parquet.py --dry-run

    # convert every report under outputs/experiments, keeping the CSVs
    .venv/bin/python scripts/reports_to_parquet.py

    # convert and remove the originals
    .venv/bin/python scripts/reports_to_parquet.py --delete-csv

    # the other direction: one Parquet report back to CSV for a spreadsheet
    .venv/bin/python scripts/reports_to_parquet.py --to-csv <file.parquet>

Column types are inferred per column, so an integer column stays integer and
a numeric column with blanks becomes a float column with nulls rather than
strings. The eval-row dumps are not touched — they are written as Parquet
from the start.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from arabic_eval.utils.io import write_failure_csv, write_report_table  # noqa: E402

PATTERNS = ("*_accuracy_failures.csv", "*_unks.csv", "intrinsic_unks.csv")


def _coerce(value: str) -> Any:
    """CSV gives strings; recover the number a column was written from."""
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def convert(path: Path, delete_csv: bool, dry_run: bool) -> Dict[str, Any]:
    out = path.with_suffix(".parquet")
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields: List[str] = list(reader.fieldnames or [])
        rows = [{k: _coerce(v) for k, v in row.items()} for row in reader]
    info = {"csv": str(path), "parquet": str(out), "rows": len(rows), "columns": len(fields)}
    if dry_run:
        return {**info, "written": False}
    write_report_table(out, rows, fields, metadata={"converted_from": path.name})
    info["bytes_csv"] = path.stat().st_size
    info["bytes_parquet"] = out.stat().st_size
    if delete_csv:
        path.unlink()
        info["csv_deleted"] = True
    return {**info, "written": True}


def to_csv(path: Path) -> Dict[str, Any]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    out = path.with_suffix(".csv")
    rows = table.to_pylist()
    flat = [
        {k: ("|".join("" if x is None else str(x) for x in v) if isinstance(v, list) else v)
         for k, v in row.items()}
        for row in rows
    ]
    n = write_failure_csv(out, flat, list(table.schema.names))
    return {"parquet": str(path), "csv": str(out), "rows": n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/experiments",
                    help="directory to walk (default: outputs/experiments)")
    ap.add_argument("--delete-csv", action="store_true",
                    help="remove each CSV once its Parquet is written")
    ap.add_argument("--dry-run", action="store_true", help="list what would be converted")
    ap.add_argument("--to-csv", metavar="FILE",
                    help="reverse: write FILE.csv from a Parquet report (UTF-8 BOM, for Excel)")
    args = ap.parse_args()

    if args.to_csv:
        info = to_csv(Path(args.to_csv))
        print(f"{info['parquet']} -> {info['csv']} ({info['rows']} rows)")
        return 0

    root = (REPO_ROOT / args.root) if not Path(args.root).is_absolute() else Path(args.root)
    found: List[Path] = []
    for pattern in PATTERNS:
        found.extend(sorted(root.glob(f"**/{pattern}")))
    if not found:
        print(f"no CSV reports under {root}")
        return 0

    total_csv = total_pq = 0
    for path in found:
        info = convert(path, args.delete_csv, args.dry_run)
        if info["written"]:
            total_csv += info["bytes_csv"]
            total_pq += info["bytes_parquet"]
            print(f"{path.relative_to(REPO_ROOT)}  {info['rows']:>7} rows  "
                  f"{info['bytes_csv']/1e6:6.2f} MB -> {info['bytes_parquet']/1e6:5.2f} MB"
                  f"{'  (csv removed)' if info.get('csv_deleted') else ''}")
        else:
            print(f"[dry-run] {path.relative_to(REPO_ROOT)}  {info['rows']} rows, {info['columns']} columns")
    if total_csv:
        print(f"\n{len(found)} report(s): {total_csv/1e6:.1f} MB of CSV -> "
              f"{total_pq/1e6:.1f} MB of Parquet ({total_csv/max(total_pq,1):.1f}x smaller)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
