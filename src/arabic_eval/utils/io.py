"""Path management, checkpoint and report-file helpers.

Row-level reports (eval-row dumps, failure reports, UNK reports) are
written as Parquet: a tenth the bytes of the equivalent CSV, loadable in
one ``pd.read_parquet`` call, and cheap to filter column-wise. The CSV
writer stays for on-demand exports of whatever a reader is looking at,
where opening the file in a spreadsheet is the point.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist, return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """Write a dict to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Dict[str, Any]:
    """Read a JSON file into a dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_failure_csv(
    path: str | Path,
    rows: Iterable[Dict[str, Any]],
    fieldnames: Sequence[str],
) -> int:
    """Write per-row records to a CSV at ``path``. Returns the rows written.

    Kept for *export* paths — the console's "export this view" button and the
    ``--csv`` flag of the report converter — where a spreadsheet is the
    destination. The pipeline itself writes Parquet via ``write_report_table``.

    Always writes the header even when ``rows`` is empty. UTF-8 with BOM so
    Excel renders Arabic correctly.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            n += 1
    return n


def _arrow_type_for(values: Iterable[Any]):
    """Narrowest Arrow type that holds every value in a report column."""
    import pyarrow as pa

    kinds = set()
    for v in values:
        if v is None:
            continue
        if isinstance(v, bool):
            kinds.add("bool")
        elif isinstance(v, int):
            kinds.add("int")
        elif isinstance(v, float):
            kinds.add("float")
        elif isinstance(v, (list, tuple)):
            kinds.add("list")
        else:
            kinds.add("str")
    if not kinds:
        return pa.string()
    if kinds == {"bool"}:
        return pa.bool_()
    if kinds == {"int"}:
        return pa.int64()
    if kinds <= {"int", "float"}:
        return pa.float64()
    if kinds == {"list"}:
        return pa.list_(pa.string())
    return pa.string()


def write_report_table(
    path: str | Path,
    rows: Iterable[Dict[str, Any]],
    fieldnames: Sequence[str],
    metadata: Dict[str, Any] | None = None,
) -> int:
    """Write per-row records to a Parquet file at ``path``. Returns rows written.

    Same contract as ``write_failure_csv`` — the caller hands dict rows plus
    the column order — but the result loads with one ``pd.read_parquet`` and
    supports column projection, so a reader can filter on a few small columns
    without decoding the Arabic text. Column types are inferred per column
    across all rows; a column whose values never appear (or that mixes types)
    becomes a string column, and a wholly empty report still writes a valid
    file with the full schema, mirroring the header-only CSV guarantee.

    ``metadata`` is stored as a JSON blob in the Parquet key-value metadata
    under ``arabic_eval``.
    """
    import json as _json

    import pyarrow as pa
    import pyarrow.parquet as pq

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    materialised = list(rows)
    names = list(fieldnames)
    columns: Dict[str, list] = {
        name: [r.get(name) for r in materialised] for name in names
    }
    fields = []
    for name in names:
        t = _arrow_type_for(columns[name])
        if t == pa.string():
            columns[name] = [
                None if v is None else (v if isinstance(v, str) else str(v))
                for v in columns[name]
            ]
        fields.append(pa.field(name, t))
    meta = None
    if metadata:
        meta = {b"arabic_eval": _json.dumps(metadata, ensure_ascii=False).encode("utf-8")}
    schema = pa.schema(fields, metadata=meta)
    table = pa.Table.from_pydict(columns, schema=schema)
    pq.write_table(table, p, compression="zstd")
    return len(materialised)
