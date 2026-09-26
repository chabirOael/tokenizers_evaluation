"""Paired statistics — the same resampling the report scripts use.

``paired_delta`` is ``freeform_judge.paired_bootstrap`` (percentile bootstrap
over items, ``np.random.RandomState(seed)``), which is what
``scripts/judge/paired_compare.py`` and ``scripts/mcq_compare.py`` call; with the
same ``n_boot`` / ``seed`` (10 000 / 0 in the report) the interval is the report's.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


def _as_dicts(a: Any, b: Any) -> Tuple[Dict[Any, float], Dict[Any, float]]:
    """Two paired samples → two dicts on shared keys (NaN pairs dropped)."""
    import pandas as pd
    if isinstance(a, dict) and isinstance(b, dict):
        da, db = a, b
    elif isinstance(a, pd.Series) and isinstance(b, pd.Series):
        j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
        da, db = j["a"].to_dict(), j["b"].to_dict()
    else:
        xa, xb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if xa.shape != xb.shape:
            raise ValueError(f"paired samples must have the same length ({xa.shape} vs {xb.shape}); "
                             "pass pandas Series to align on their index")
        ok = ~(np.isnan(xa) | np.isnan(xb))
        da = {i: float(v) for i, v in enumerate(xa) if ok[i]}
        db = {i: float(v) for i, v in enumerate(xb) if ok[i]}
    da = {k: float(v) for k, v in da.items() if v is not None and not _isnan(v)}
    db = {k: float(v) for k, v in db.items() if v is not None and not _isnan(v)}
    return da, db


def _isnan(v: Any) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def paired_delta(a: Any, b: Any, *, n_boot: int = 10000, seed: int = 0) -> Dict[str, Any]:
    """Mean of **a − b** over paired items, with a 95 % percentile-bootstrap CI.

    ``a`` / ``b``: dicts keyed by item id (shared ids used), pandas Series (aligned
    on the index, NaN pairs dropped) or equal-length arrays (booleans count as 0/1,
    so accuracy differences work). Returns n, mean_a, mean_b, delta, ci_low,
    ci_high, and the win / tie / loss rates of a against b."""
    from arabic_eval.judge.freeform_judge import paired_bootstrap
    da, db = _as_dicts(a, b)
    r = paired_bootstrap(db, da, n_boot=n_boot, seed=seed)       # other − base = a − b
    ids = sorted(set(da) & set(db), key=str)
    return {
        "n": r["n"],
        "mean_a": round(float(np.mean([da[i] for i in ids])), 6) if ids else None,
        "mean_b": round(float(np.mean([db[i] for i in ids])), 6) if ids else None,
        "delta": r["delta_mean"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
        "win_rate": r["win_rate"], "tie_rate": r["tie_rate"], "loss_rate": r["loss_rate"],
        "n_boot": n_boot, "seed": seed,
    }


def mcnemar(a: Any, b: Any) -> Dict[str, Any]:
    """Exact McNemar test on two paired correctness vectors (same rows, same order):
    rows only a got right, rows only b got right, two-sided exact binomial p."""
    xa, xb = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if xa.shape != xb.shape:
        raise ValueError("mcnemar needs two vectors over the same rows")
    only_a, only_b = int((xa & ~xb).sum()), int((~xa & xb).sum())
    n = only_a + only_b
    p = None
    if n:
        k = min(only_a, only_b)
        tail = sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + n * math.log(0.5))
                   for i in range(k + 1))
        p = round(min(1.0, 2.0 * tail), 6)
    return {"n": int(len(xa)), "only_a": only_a, "only_b": only_b, "p": p}


def bootstrap_ci(x: Any, stat: Callable[[np.ndarray], float] = np.mean, *, n_boot: int = 10000,
                 seed: int = 0, level: float = 0.95) -> Dict[str, Optional[float]]:
    """Percentile bootstrap CI of a statistic of one sample (NaN dropped)."""
    v = np.asarray(x, dtype=float)
    v = v[~np.isnan(v)]
    if not len(v):
        return {"n": 0, "value": None, "ci_low": None, "ci_high": None}
    rng = np.random.RandomState(seed)
    boots = np.array([stat(v[rng.randint(0, len(v), len(v))]) for _ in range(n_boot)])
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {"n": int(len(v)), "value": round(float(stat(v)), 6),
            "ci_low": round(float(np.percentile(boots, lo)), 6), "ci_high": round(float(np.percentile(boots, hi)), 6)}
