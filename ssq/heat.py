"""号码热度：热度指数与组合结构特征

热度指数 H = 实际中奖注数 / (销量/2 × 理论概率)，H=1 表示开出的这组号
被购买的程度与均匀随机一致，H>1 表示彩民买它买得更多。
"""
from collections import Counter
from math import comb

import numpy as np

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