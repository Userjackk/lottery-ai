"""号码热度：热度指数与组合结构特征

热度指数 H = 实际中奖注数 / (销量/2 × 理论概率)，H=1 表示开出的这组号
被购买的程度与均匀随机一致，H>1 表示彩民买它买得更多。
"""
from collections import Counter
from math import comb

import numpy as np
from scipy import stats

from .config import RED_MAX, BLUE_MAX, RED_COUNT

_TOTAL_COMBOS = comb(RED_MAX, RED_COUNT) * BLUE_MAX
P1 = 1 / _TOTAL_COMBOS          # 一等奖概率
P2 = 15 / _TOTAL_COMBOS         # 二等奖概率（6 红 0 蓝）

FEATURE_KEYS = [
    "small_count", "date_count", "sum_value", "span", "longest_run",
    "consec_pairs", "same_tail", "odd_count", "zone_balance",
    "has_arith3", "blue_small", "repeat_prev",
]


def heat_index(n_prize: int | None, sales: int, p_win: float) -> float:
    """热度指数；缺中奖注数或销量无效时返回 nan"""
    if n_prize is None or sales is None or sales <= 0 or p_win <= 0:
        return float("nan")
    return n_prize / (sales / 2 * p_win)


def has_arithmetic_pattern(reds: list[int]) -> bool:
    """是否存在 3 项及以上等差数列

    与 ssq/models/ensemble.py 里同名逻辑一致，刻意重复以避免分析层耦合模型层。
    """
    rs = set(reds)
    for a in reds:
        for step in range(1, 17):
            if a + step in rs and a + 2 * step in rs:
                return True
    return False


def combo_features(reds: list[int], blue: int,
                   prev_reds: list[int] | None = None) -> dict[str, float]:
    """单期开奖组合的结构特征"""
    rs = sorted(reds)

    longest_run = 1
    run = 1
    for i in range(1, len(rs)):
        if rs[i] == rs[i - 1] + 1:
            run += 1
            longest_run = max(longest_run, run)
        else:
            run = 1

    tail_counts = Counter(r % 10 for r in rs)
    zones = [sum(1 for r in rs if lo <= r <= hi)
             for lo, hi in ((1, 11), (12, 22), (23, 33))]

    return {
        "small_count": float(sum(1 for r in rs if r <= 12)),
        "date_count": float(sum(1 for r in rs if r <= 31)),
        "sum_value": float(sum(rs)),
        "span": float(rs[-1] - rs[0]),
        "longest_run": float(longest_run),
        "consec_pairs": float(sum(1 for i in range(1, len(rs))
                                  if rs[i] == rs[i - 1] + 1)),
        "same_tail": float(sum(comb(c, 2) for c in tail_counts.values())),
        "odd_count": float(sum(1 for r in rs if r % 2 == 1)),
        "zone_balance": float(np.std(zones)),
        "has_arith3": 1.0 if has_arithmetic_pattern(rs) else 0.0,
        "blue_small": 1.0 if blue <= 12 else 0.0,
        "repeat_prev": float(len(set(rs) & set(prev_reds))) if prev_reds else 0.0,
    }


def ols_hc1(X: np.ndarray, y: np.ndarray) -> dict:
    """OLS + HC1 稳健标准误；自动前置截距。

    方差为 0 的列不参与回归，其 beta 记 0、se/t/p 记 nan，不抛异常。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape

    valid = np.std(X, axis=0) > 0
    Xv = np.column_stack([np.ones(n), X[:, valid]])
    k_v = Xv.shape[1]

    xtx_inv = np.linalg.pinv(Xv.T @ Xv)
    beta_v = xtx_inv @ Xv.T @ y
    resid = y - Xv @ beta_v
    meat = (Xv * resid[:, None] ** 2).T @ Xv
    cov = xtx_inv @ meat @ xtx_inv * n / max(n - k_v, 1)

    se_v = np.sqrt(np.diag(cov))
    t_v = np.divide(beta_v, se_v,
                    out=np.full_like(beta_v, np.nan), where=se_v > 0)
    with np.errstate(invalid="ignore"):
        p_v = 2 * stats.t.sf(np.abs(t_v), df=n - k_v)

    beta = np.zeros(k + 1)
    se = np.full(k + 1, np.nan)
    t = np.full(k + 1, np.nan)
    p = np.full(k + 1, np.nan)
    beta[0], se[0], t[0], p[0] = beta_v[0], se_v[0], t_v[0], p_v[0]
    idx = np.where(valid)[0] + 1
    beta[idx], se[idx], t[idx], p[idx] = beta_v[1:], se_v[1:], t_v[1:], p_v[1:]
    return {"beta": beta, "se": se, "t": t, "p": p}


def bh_correct(pvals: list[float]) -> np.ndarray:
    """Benjamini-Hochberg 调整后 p 值；nan 原样透传且不参与排名"""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return out

    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(ranked, 1.0)
    out[ok] = adj
    return out


def quantile_groups(values: np.ndarray, n_groups: int = 3) -> np.ndarray:
    """按分位数分档，返回 0..n_groups-1 的标签"""
    v = np.asarray(values, dtype=float)
    edges = np.quantile(v, np.arange(1, n_groups) / n_groups)
    return np.searchsorted(edges, v, side="right").astype(int)


def bootstrap_ci(values: np.ndarray, stat=np.mean, n_boot: int = 2000,
                 seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    """有放回 bootstrap 置信区间，nan 先剔除"""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = np.array([stat(v[i]) for i in idx])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def build_dataset(prize_rows: list[dict], draws: list[dict]) -> list[dict]:
    """按 draw_no 内连接奖级数据与开奖数据，计算热度指数与组合特征"""
    prize_by_no = {r["draw_no"]: r for r in prize_rows if r.get("sales")}
    draw_by_no = {d["draw_no"]: d for d in draws}
    common = sorted(set(prize_by_no) & set(draw_by_no))

    ordered = sorted(draw_by_no)
    prev_of = {no: (draw_by_no[ordered[i - 1]] if i > 0 else None)
               for i, no in enumerate(ordered)}

    rows = []
    for no in common:
        p = prize_by_no[no]
        d = draw_by_no[no]
        reds = [d[f"red{i}"] for i in range(1, 7)]
        prev = prev_of[no]
        prev_reds = [prev[f"red{i}"] for i in range(1, 7)] if prev else None

        row = {
            "draw_no": no,
            "date": d.get("date", ""),
            "sales": p["sales"],
            "n1": p.get("n1"),
            "n2": p.get("n2"),
            "h1": heat_index(p.get("n1"), p["sales"], P1),
            "h2": heat_index(p.get("n2"), p["sales"], P2),
        }
        row.update(combo_features(reds, d["blue"], prev_reds))
        rows.append(row)
    return rows


def check_heat_uniformity(h2: np.ndarray, tolerance: float = 0.15) -> dict:
    """均匀性自检：全体 H2 的均值应接近 1，否则说明口径有误"""
    v = np.asarray(h2, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"mean": float("nan"), "n": 0, "ok": False}
    mean = float(v.mean())
    return {"mean": mean, "n": int(len(v)), "ok": bool(abs(mean - 1) <= tolerance)}