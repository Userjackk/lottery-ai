from collections import Counter
from .config import RED_MAX, BLUE_MAX


def _reds(draw: dict) -> list[int]:
    return [draw["red1"], draw["red2"], draw["red3"], draw["red4"], draw["red5"], draw["red6"]]


def frequency_analysis(draws: list[dict], window: int = 30) -> dict:
    """统计各号码在最近 window 期内的出现频率"""
    recent = draws[-window:] if len(draws) > window else draws
    red_counter = Counter()
    blue_counter = Counter()
    for d in recent:
        for r in _reds(d):
            red_counter[r] += 1
        blue_counter[d["blue"]] += 1
    n = len(recent)
    return {
        "red": {i: red_counter.get(i, 0) / n for i in range(1, RED_MAX + 1)},
        "blue": {i: blue_counter.get(i, 0) / n for i in range(1, BLUE_MAX + 1)},
        "window": n,
    }


def omission_analysis(draws: list[dict]) -> dict:
    """各号码的遗漏值（距上次出现过了多少期）"""
    red_omit = {i: len(draws) for i in range(1, RED_MAX + 1)}
    blue_omit = {i: len(draws) for i in range(1, BLUE_MAX + 1)}
    for idx, d in enumerate(draws):
        for r in _reds(d):
            red_omit[r] = len(draws) - 1 - idx
        blue_omit[d["blue"]] = len(draws) - 1 - idx
    return {"red": red_omit, "blue": blue_omit}


def hot_cold_analysis(draws: list[dict], hot_window: int = 30, cold_window: int = 100) -> dict:
    """冷热号分类：热号=近30期高频，冷号=遗漏>阈值"""
    freq = frequency_analysis(draws, hot_window)
    omit = omission_analysis(draws)
    avg_freq = 6 * hot_window / RED_MAX  # 理论平均出现次数
    hot_threshold = avg_freq * 1.5
    cold_threshold = cold_window * 0.15

    result = {"hot": [], "warm": [], "cold": []}
    for i in range(1, RED_MAX + 1):
        f = freq["red"][i] * hot_window
        o = omit["red"][i]
        if f >= hot_threshold:
            result["hot"].append(i)
        elif o >= cold_threshold:
            result["cold"].append(i)
        else:
            result["warm"].append(i)
    return result


def pattern_analysis(draws: list[dict], window: int = 30) -> dict:
    """奇偶比、大小比、三区分布"""
    recent = draws[-window:] if len(draws) > window else draws
    results = []
    for d in recent:
        reds = _reds(d)
        odd = sum(1 for r in reds if r % 2 == 1)
        big = sum(1 for r in reds if r > 16)
        z1 = sum(1 for r in reds if r <= 11)
        z2 = sum(1 for r in reds if 12 <= r <= 22)
        z3 = sum(1 for r in reds if r >= 23)
        results.append({
            "odd_even": f"{odd}:{6 - odd}",
            "big_small": f"{big}:{6 - big}",
            "zones": [z1, z2, z3],
        })
    # 汇总
    odd_ratios = Counter(r["odd_even"] for r in results)
    big_ratios = Counter(r["big_small"] for r in results)
    return {
        "odd_even_dist": dict(odd_ratios.most_common(5)),
        "big_small_dist": dict(big_ratios.most_common(5)),
        "recent_patterns": results[-5:],
        "window": len(recent),
    }


def sum_analysis(draws: list[dict], window: int = 30) -> dict:
    """红球和值分析"""
    recent = draws[-window:] if len(draws) > window else draws
    sums = [sum(_reds(d)) for d in recent]
    return {
        "min": min(sums),
        "max": max(sums),
        "avg": sum(sums) / len(sums),
        "recent": sums[-5:],
    }


def consecutive_analysis(draws: list[dict], window: int = 30) -> dict:
    """连号出现频率"""
    recent = draws[-window:] if len(draws) > window else draws
    counts = []
    for d in recent:
        reds = _reds(d)
        consecutive = sum(1 for i in range(len(reds) - 1) if reds[i + 1] - reds[i] == 1)
        counts.append(consecutive)
    has_consecutive = sum(1 for c in counts if c > 0)
    return {
        "frequency": has_consecutive / len(recent),
        "avg_pairs": sum(counts) / len(counts),
        "recent": counts[-5:],
    }
