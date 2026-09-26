"""``show()``: hand a figure or a table to the Analysis tab — and the chart style.

Inside the console's analysis worker, ``show`` writes the object under the
session folder (Plotly → JSON drawn interactively by the page, which shapes
Arabic text correctly; matplotlib → PNG; DataFrame → CSV + a preview) and
reports it to the worker. Anywhere else it falls back to IPython's display or
a printed summary, so notebooks and scripts can use the same code.

Colours are the dataviz reference palette (validated categorical order, blue
sequential ramp, blue↔red diverging with a gray midpoint), thin marks,
recessive hairline grid.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# --------------------------------------------------------------------------
# palette (dataviz reference instance, light mode — the console is light)
# --------------------------------------------------------------------------

PALETTE: List[str] = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL: List[str] = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
                         "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
#: Blue arm = steps 650 / 450 / 250 of the reference blue ramp; red arm = the same OKLab
#: lightness and chroma at the reference red's hue (derived 2026-09-25, both arms pass the
#: validator's --ordinal checks); gray midpoint #f0efec.
DIVERGING: List[str] = ["#104281", "#2a78d6", "#86b6ef", "#f0efec", "#ea9a93", "#c74845", "#762221"]
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
FONT = "IBM Plex Sans, system-ui, -apple-system, Segoe UI, sans-serif"

_INSTALLED = False


def plotly_template():
    import plotly.graph_objects as go
    axis = dict(gridcolor=GRID, gridwidth=1, linecolor=BASELINE, showline=True, zeroline=False, ticks="",
                tickfont=dict(color=INK_2), title=dict(font=dict(color=INK_2)), automargin=True)
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=FONT, color=INK, size=13), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        colorway=PALETTE, title=dict(x=0, xanchor="left", font=dict(size=15, color=INK)),
        xaxis=axis, yaxis=axis, legend=dict(font=dict(color=INK_2), bgcolor="rgba(0,0,0,0)"),
        hovermode="closest", hoverlabel=dict(bgcolor="#ffffff", font=dict(family=FONT, color=INK)),
        margin=dict(l=60, r=30, t=60, b=50), bargap=0.35, barcornerradius=4,
        colorscale=dict(sequential=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
                        diverging=[[i / (len(DIVERGING) - 1), c] for i, c in enumerate(DIVERGING)]),
    )
    t.data.scatter = [go.Scatter(line=dict(width=2), marker=dict(size=8, line=dict(width=2, color=SURFACE)))]
    t.data.bar = [go.Bar(marker=dict(line=dict(width=0)))]
    return t


def install_style() -> None:
    """Register the Plotly template as the default and apply the same palette to matplotlib."""
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        import plotly.io as pio
        pio.templates["arabic_eval"] = plotly_template()
        pio.templates.default = "arabic_eval"
    except ImportError:
        pass
    try:
        import matplotlib
        from cycler import cycler
        matplotlib.rcParams.update({
            "axes.prop_cycle": cycler(color=PALETTE), "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
            "axes.edgecolor": BASELINE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1,
            "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
            "lines.linewidth": 2, "lines.markersize": 6, "axes.spines.top": False, "axes.spines.right": False,
            "font.family": "sans-serif", "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans", "Arial"],
            "figure.dpi": 110, "savefig.dpi": 144,
        })
    except ImportError:
        pass
    _INSTALLED = True


# --------------------------------------------------------------------------
# the sink (set by the analysis worker around each step)
# --------------------------------------------------------------------------

_SINK: Optional[Callable[[Dict[str, Any]], None]] = None
_OUT_DIR: Optional[Path] = None
_PREFIX = "out"
_COUNT = itertools.count(1)
MAX_TABLE_PREVIEW = 200


def set_sink(sink: Optional[Callable[[Dict[str, Any]], None]], out_dir: Optional[Path] = None, prefix: str = "out") -> None:
    global _SINK, _OUT_DIR, _PREFIX, _COUNT
    _SINK, _OUT_DIR, _PREFIX = sink, (Path(out_dir) if out_dir else None), prefix
    _COUNT = itertools.count(1)


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return None if math.isnan(v) or math.isinf(v) else v
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return _jsonable(v.item())
        if isinstance(v, np.ndarray):
            return str(v.tolist())[:200]
    except ImportError:  # pragma: no cover
        pass
    return str(v)[:500]


def _table_payload(df, title: Optional[str], max_rows: int) -> Dict[str, Any]:
    import pandas as pd
    if isinstance(df, pd.Series):
        df = df.to_frame(df.name if df.name is not None else "value")
    # an unnamed integer index (a RangeIndex, or one a sort reshuffled) carries no information: leave it out
    plain = df.index.nlevels == 1 and df.index.name is None and pd.api.types.is_integer_dtype(df.index.dtype)
    shown = df if plain else df.reset_index()
    head = shown.head(max_rows)
    cols = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in head.columns]
    rows = [[_jsonable(v) for v in r] for r in head.itertuples(index=False, name=None)]
    with pd.option_context("display.max_columns", 14, "display.width", 180, "display.max_colwidth", 60):
        text = df.to_string(max_rows=20)
    return {"kind": "table", "title": title, "columns": cols, "rows": rows, "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]), "text": text}


def _plotly_summary(fig) -> str:
    parts = []
    for tr in fig.data[:12]:
        n = None
        for attr in ("x", "y", "values", "z"):
            v = getattr(tr, attr, None)
            if v is not None:
                try:
                    n = len(v)
                except TypeError:
                    n = None
                break
        parts.append(f"{tr.type}{'(' + str(tr.name) + ')' if tr.name else ''}{'[' + str(n) + ']' if n is not None else ''}")
    lay = fig.layout
    xt = lay.xaxis.title.text if lay.xaxis and lay.xaxis.title else None
    yt = lay.yaxis.title.text if lay.yaxis and lay.yaxis.title else None
    return f"traces: {', '.join(parts) or 'none'}" + (f"; x: {xt}" if xt else "") + (f"; y: {yt}" if yt else "")


def show(obj: Any, title: Optional[str] = None, *, max_rows: int = MAX_TABLE_PREVIEW) -> None:
    """Display a Plotly figure, a matplotlib figure, a pandas DataFrame / Series or a markdown string in the Analysis tab.

    Prefer Plotly for charts (``import plotly.express as px``): it is interactive and
    renders Arabic labels correctly (matplotlib does not shape Arabic). Give every
    figure a title and axis titles. Tables show up to ``max_rows`` rows; the full
    table is saved as CSV next to it."""
    install_style()
    payload: Dict[str, Any]
    kind = type(obj).__module__ + "." + type(obj).__name__
    n = next(_COUNT)
    stem = f"{_PREFIX}_{n}"
    out_dir = _OUT_DIR
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    if kind.startswith("plotly.") and hasattr(obj, "to_json"):
        if title and not (obj.layout.title and obj.layout.title.text):
            obj.update_layout(title=title)
        title = title or (obj.layout.title.text if obj.layout.title else None)
        payload = {"kind": "plotly", "title": title, "text": f"[plotly figure '{title or 'untitled'}': {_plotly_summary(obj)}]"}
        if out_dir is not None:
            p = out_dir / f"{stem}.plotly.json"
            p.write_text(obj.to_json(), encoding="utf-8")
            payload.update(path=p.name, bytes=p.stat().st_size)
    elif kind.startswith("matplotlib.") and hasattr(obj, "savefig"):
        payload = {"kind": "image", "title": title, "format": "png", "text": f"[matplotlib figure '{title or 'untitled'}']"}
        if out_dir is not None:
            p = out_dir / f"{stem}.png"
            obj.savefig(p, bbox_inches="tight")
            payload.update(path=p.name, bytes=p.stat().st_size)
        try:
            import matplotlib.pyplot as plt
            plt.close(obj)
        except ImportError:  # pragma: no cover
            pass
    elif kind.startswith("pandas.") and hasattr(obj, "to_csv"):
        payload = _table_payload(obj, title, max_rows)
        if out_dir is not None:
            p = out_dir / f"{stem}.csv"
            obj.to_csv(p)
            payload.update(path=p.name)
    elif isinstance(obj, str):
        payload = {"kind": "markdown", "title": title, "markdown": obj, "text": obj[:2000]}
    else:
        raise TypeError(f"show() takes a Plotly / matplotlib figure, a DataFrame / Series or a string, not {kind}")

    if _SINK is not None:
        _SINK(payload)
        return
    try:                                         # a notebook
        from IPython import get_ipython
        if get_ipython() is not None:
            from IPython.display import display
            display(obj)
            return
    except ImportError:
        pass
    print(payload.get("text") or json.dumps({k: v for k, v in payload.items() if k != "rows"})[:2000])
