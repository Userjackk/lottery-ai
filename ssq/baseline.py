"""随机基线：用于和模型回测结果做对照"""
import numpy as np
from .config import RED_MAX, BLUE_MAX, RED_COUNT


def _reds(d):
    return [d["red1"], d["red2"], d["red3"], d["red4"], d["red5"], d["red6"]]


def random_baseline_backtest(draws, start=100, end=None, seed=42):
    """随机选号滚动回测（对照组）

    每期随机选 6 个红球 + 1 个蓝球，统计命中分布。
    返回 dict，包含与 model backtest 可对比的指标。
    """
    rng = np.random.default_rng(seed)
    if end is None:
        end = len(draws)
    total = end - start

    red_hit_dist = {i: 0 for i in range(7)}
    blue_hits = 0

    for i in range(start, end):
        actual = draws[i]
        actual_reds = set(_reds(actual))

        pred_reds = set(rng.choice(range(1, RED_MAX + 1), size=RED_COUNT, replace=False))
        pred_blue = int(rng.integers(1, BLUE_MAX + 1))

        rh = len(pred_reds & actual_reds)
        bh = 1 if pred_blue == actual["blue"] else 0

        red_hit_dist[rh] += 1
        blue_hits += bh

    avg_red = sum(k * v for k, v in red_hit_dist.items()) / total
    avg_blue_pct = blue_hits / total * 100

    return {
        "red_hit_dist": red_hit_dist,
        "blue_hits": blue_hits,
        "total": total,
        "avg_red": avg_red,
        "avg_blue_pct": avg_blue_pct,
    }


def format_random_baseline(results: dict) -> str:
    """格式化随机基线回测结果"""
    total = results["total"]
    lines = []
    lines.append("\n" + "=" * 55)
    lines.append("【随机基线对照组】")
    lines.append("=" * 55)
    lines.append(f"总期数: {total}")
    lines.append("\n红球命中分布:")
    for hits, count in results["red_hit_dist"].items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        lines.append(f"  {hits}球: {count:4d}期 ({pct:5.1f}%) {bar}")
    lines.append(f"\n红球平均命中: {results['avg_red']:.2f}/6 (随机期望: {6*6/33:.2f})")
    lines.append(f"蓝球命中率: {results['avg_blue_pct']:.1f}% (随机期望: {100/BLUE_MAX:.1f}%)")
    return "\n".join(lines)
