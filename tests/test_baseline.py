"""测试随机基线对比回测"""
import numpy as np
from ssq.baseline import random_baseline_backtest


def _make_draw(idx):
    np.random.seed(idx)
    red = sorted(np.random.choice(range(1, 34), size=6, replace=False))
    blue = int(np.random.choice(range(1, 17)))
    return {
        "draw_no": f"2026{idx:03d}",
        "date": "2026-01-01",
        "red1": red[0], "red2": red[1], "red3": red[2],
        "red4": red[3], "red5": red[4], "red6": red[5],
        "blue": blue,
    }


def test_random_baseline_returns_results():
    """随机基线回测应返回可统计的结果"""
    draws = [_make_draw(i) for i in range(150)]
    results = random_baseline_backtest(draws, start=100)
    assert "red_hit_dist" in results
    assert "blue_hits" in results
    assert "total" in results
    assert "avg_red" in results
    assert "avg_blue_pct" in results


def test_random_baseline_expected_values():
    """随机基线红球均值应接近理论期望"""
    np.random.seed(42)
    draws = [_make_draw(i) for i in range(300)]
    results = random_baseline_backtest(draws, start=100)
    # 理论期望 6*6/33 ≈ 1.09，大量样本下应接近
    assert abs(results["avg_red"] - 1.09) < 0.3
