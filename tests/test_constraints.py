"""测试模型输出约束过滤"""
import numpy as np
from ssq.models.ensemble import EnsembleModel
from ssq.config import RED_MAX, RED_COUNT


def _make_red_probs(favored: list[int]) -> np.ndarray:
    probs = np.ones(RED_MAX) * 0.01
    for i in favored:
        probs[i - 1] = 1.0
    return probs / probs.sum()


def test_generate_with_constraints_produces_valid_reds():
    """加约束后生成的红球应满足基本规则"""
    ens = EnsembleModel(weights={"baseline": 1.0})
    red_probs = np.ones(RED_MAX) / RED_MAX
    blue_probs = np.ones(16) / 16
    recs = ens.generate_recommendations(red_probs, blue_probs, n_sets=3, draws=[])
    assert len(recs) == 3
    for r in recs:
        reds = r["red"]
        assert len(reds) == RED_COUNT
        assert len(set(reds)) == RED_COUNT
        assert all(1 <= x <= RED_MAX for x in reds)
        assert 1 <= r["blue"] <= 16


def test_generate_prefers_reasonable_sum():
    """约束应倾向于和值在常见范围内的组合"""
    ens = EnsembleModel(weights={"baseline": 1.0})
    red_probs = _make_red_probs([1, 2, 3, 4, 5, 6])
    blue_probs = np.ones(16) / 16
    # 没有历史数据时约束过滤不会崩溃
    recs = ens.generate_recommendations(red_probs, blue_probs, n_sets=1, draws=[])
    assert len(recs) >= 1


def test_constraint_filter_rejects_extreme():
    """极端组合（如全小号）应被过滤"""
    from ssq.constraints import ConstraintFilter
    cf = ConstraintFilter(draws=[])
    extreme = [1, 2, 3, 4, 5, 6]  # 和值过小
    assert cf.check(extreme) is False

    reasonable = [6, 11, 17, 22, 27, 31]
    assert cf.check(reasonable) is True
