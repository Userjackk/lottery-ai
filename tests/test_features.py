"""测试特征工程扩展"""
import numpy as np
from ssq.features import build_feature_matrix, build_next_features


def _make_draw(red, blue, draw_no="2026001"):
    r = sorted(red)
    return {
        "draw_no": draw_no,
        "date": "2026-01-01",
        "red1": r[0], "red2": r[1], "red3": r[2],
        "red4": r[3], "red5": r[4], "red6": r[5],
        "blue": blue,
    }


def test_feature_matrix_shape():
    """特征矩阵形状应正确，且包含扩展特征"""
    draws = [_make_draw([1, 2, 3, 4, 5, 6], 7, f"2026{i:03d}") for i in range(1, 80)]
    X, y_red, y_blue = build_feature_matrix(draws)
    assert X.ndim == 2
    assert y_red.shape[1] == 33
    assert y_blue.shape[1] == 16
    # 扩展后特征维度应比 README 里写的 203 大
    assert X.shape[1] > 203


def test_span_feature():
    """跨度特征应反映最大最小红球差"""
    draws = [_make_draw([1, 2, 3, 4, 5, 33], 7, f"2026{i:03d}") for i in range(1, 80)]
    X, _, _ = build_feature_matrix(draws)
    # 最后两列是 跨度、AC值
    span_values = X[:, -2]
    expected = 32 / 33
    assert abs(span_values[0] - expected) < 1e-6


def test_ac_value_feature():
    """AC值特征应为非负数"""
    draws = [_make_draw([1, 2, 3, 4, 5, 6], 7, f"2026{i:03d}") for i in range(1, 80)]
    X, _, _ = build_feature_matrix(draws)
    ac_values = X[:, -1]
    assert (ac_values >= 0).all()


def test_next_features_shape():
    """下一期特征应为1行"""
    draws = [_make_draw([1, 2, 3, 4, 5, 6], 7, f"2026{i:03d}") for i in range(1, 80)]
    X_next = build_next_features(draws)
    assert X_next.shape[0] == 1
