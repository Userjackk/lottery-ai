"""测试集成模型权重可配置"""
import numpy as np
from ssq.models.ensemble import EnsembleModel


def test_ensemble_weights_override():
    """应能通过构造函数覆盖默认权重"""
    weights = {"baseline": 0.5, "xgboost": 0.5}
    ens = EnsembleModel(weights)
    assert ens.weights == weights


def test_ensemble_combine_shape():
    """融合后概率向量形状正确"""
    ens = EnsembleModel({"baseline": 0.5, "xgboost": 0.5})
    predictions = {
        "baseline": (np.ones(33) / 33, np.ones(16) / 16),
        "xgboost": (np.ones(33) / 33, np.ones(16) / 16),
    }
    red, blue = ens.combine(predictions)
    assert red.shape == (33,)
    assert blue.shape == (16,)
    np.testing.assert_allclose(red.sum(), 1.0, atol=1e-6)
    np.testing.assert_allclose(blue.sum(), 1.0, atol=1e-6)


def test_ensemble_recommendations_range():
    """所有推荐方案的红球必须在 1-33 范围内，蓝球在 1-16 范围内"""
    ens = EnsembleModel()
    red_probs = np.linspace(0.01, 0.05, 33)
    red_probs /= red_probs.sum()
    blue_probs = np.linspace(0.01, 0.1, 16)
    blue_probs /= blue_probs.sum()

    for _ in range(20):
        recs = ens.generate_recommendations(red_probs, blue_probs, n_sets=3)
        assert len(recs) == 3
        for rec in recs:
            assert len(rec["red"]) == 6
            assert all(1 <= r <= 33 for r in rec["red"]), f"红球超出范围: {rec['red']}"
            assert 1 <= rec["blue"] <= 16, f"蓝球超出范围: {rec['blue']}"

