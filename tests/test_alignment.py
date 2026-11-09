"""测试预测/回测对齐契约：特征到第 D 期 → 预测第 D+1 期，生产模型用全量数据"""
import numpy as np

from ssq.features import build_feature_matrix, build_next_features
from ssq.models.lstm_model import build_lstm_input
from ssq.models.xgb_model import XGBPredictor


def _make_draw(red, blue, draw_no="2026001"):
    r = sorted(red)
    return {
        "draw_no": draw_no,
        "date": "2026-01-01",
        "red1": r[0], "red2": r[1], "red3": r[2],
        "red4": r[3], "red5": r[4], "red6": r[5],
        "blue": blue,
    }


def _synthetic_draws(n, seed=7):
    rng = np.random.default_rng(seed)
    draws = []
    for i in range(n):
        red = sorted(rng.choice(range(1, 34), size=6, replace=False).tolist())
        draws.append(_make_draw(red, int(rng.integers(1, 17)), f"2026{i+1:03d}"))
    return draws


def test_lstm_input_seq_ends_at_latest():
    """LSTM 输入序列应以下一期特征结尾（而非特征矩阵最后一行）"""
    draws = _synthetic_draws(80)
    X, _, _ = build_feature_matrix(draws)
    next_feat = build_next_features(draws)
    seq = build_lstm_input(X, draws, seq_len=20)
    assert seq.shape == (20, X.shape[1])
    # 最后一行必须是下一期特征（最新一期），而不是 X 的最后一行
    np.testing.assert_array_equal(seq[-1], next_feat[0])


def test_predict_probs_target_not_leaked_and_aligned():
    """预测第 i 期：只改目标期开奖不得影响预测（无泄漏），改训练区必须影响预测（对齐）"""
    from backtest import _predict_probs

    draws = _synthetic_draws(110)

    preds_ref = _predict_probs(draws, 105, include_lstm=False)
    p_ref_red = preds_ref["xgboost"][0]

    # 无泄漏：只改目标期（draws[105]），预测必须完全不变
    draws_leak = _synthetic_draws(110)
    draws_leak[105] = _make_draw([1, 2, 3, 4, 5, 6], 9, draws_leak[105]["draw_no"])
    preds_leak = _predict_probs(draws_leak, 105, include_lstm=False)
    np.testing.assert_allclose(preds_leak["xgboost"][0], p_ref_red, atol=1e-10)

    # 对齐：改训练区最新 15 期（draws[90:105]），预测必须改变
    draws_mod = _synthetic_draws(110)
    for k in range(90, 105):
        draws_mod[k] = _make_draw([1, 2, 3, 4, 5, 6], 9, draws_mod[k]["draw_no"])
    preds_mod = _predict_probs(draws_mod, 105, include_lstm=False)
    assert not np.allclose(preds_mod["xgboost"][0], p_ref_red, atol=1e-9)


def test_train_uses_full_data_for_production(monkeypatch, tmp_path):
    """生产模型应在全部数据上训练（回测评估模型才用扣除 holdout 的数据）"""
    from ssq.predictor import Predictor
    from ssq import db as ssq_db

    monkeypatch.setattr(ssq_db, "DB_PATH", tmp_path / "test.db")
    ssq_db.init_db()

    draws = _synthetic_draws(80)
    X_full, _, _ = build_feature_matrix(draws)

    seen_sizes = []
    orig_fit = XGBPredictor.fit
    monkeypatch.setattr(XGBPredictor, "fit",
                        lambda self, X, yr, yb: (seen_sizes.append(len(X)), orig_fit(self, X, yr, yb))[1])

    p = Predictor(draws)
    p.train()

    assert X_full.shape[0] in seen_sizes, f"生产训练应包含全量 {X_full.shape[0]} 行，实际 fit 尺寸: {seen_sizes}"


def test_unavailable_lstm_excluded_from_ensemble():
    """LSTM 不可用（未训练/torch缺失）时不应以 uniform 概率混入融合"""
    from ssq.predictor import Predictor

    draws = _synthetic_draws(80)
    p = Predictor(draws)
    p._get_features()
    p.baseline.fit(draws)
    p.lstm.model = None  # 模拟 torch 缺失 / 未训练

    preds = p._collect_predictions(build_next_features(draws))
    assert "lstm" not in preds
    assert "baseline" in preds


def test_available_lstm_included():
    """LSTM 已训练时应出现在融合输入里"""
    from ssq.predictor import Predictor

    class _FakeLstm:
        model = object()
        seq_len = 20
        _available = True

        def predict_proba(self, X):
            return np.ones(33) / 33, np.ones(16) / 16

    draws = _synthetic_draws(80)
    p = Predictor(draws)
    p._get_features()
    p.baseline.fit(draws)
    p.lstm = _FakeLstm()

    preds = p._collect_predictions(build_next_features(draws))
    assert "lstm" in preds


def test_model_cache_keys_not_legacy(monkeypatch, tmp_path):
    """模型缓存键必须区别于旧口径的 'lstm'/'xgb'（旧口径缓存不得被复用）"""
    from ssq.predictor import LSTM_CACHE_KEY, XGB_CACHE_KEY
    from ssq import db as ssq_db

    assert LSTM_CACHE_KEY != "lstm"
    assert XGB_CACHE_KEY != "xgb"
    monkeypatch.setattr(ssq_db, "DB_PATH", tmp_path / "test.db")
    ssq_db.init_db()
    ssq_db.save_model("lstm", "2026001", b"legacy")
    ssq_db.save_model("xgb", "2026001", b"legacy")
    assert ssq_db.load_model(LSTM_CACHE_KEY) is None
    assert ssq_db.load_model(XGB_CACHE_KEY) is None


def test_walk_forward_dumps_per_model_probs(tmp_path):
    """滚动回测应导出各模型逐期概率，供离线调融合权重"""
    from backtest import walk_forward_backtest

    draws = _synthetic_draws(60)
    dump = tmp_path / "probs.npz"
    walk_forward_backtest(draws, start=55, end=57, include_lstm=False, dump_path=str(dump))

    z = np.load(dump, allow_pickle=False)
    assert set(z.files) >= {"model_names", "red_probs", "blue_probs", "actual_red", "actual_blue"}
    # 60 期数据 X 行数 < 50，只有 baseline
    assert list(z["model_names"]) == ["baseline"]
    assert z["red_probs"].shape == (2, 1, 33)
    assert z["blue_probs"].shape == (2, 1, 16)
    assert z["actual_red"].shape == (2, 33)
    assert z["actual_blue"].shape == (2,)


def test_walk_forward_hit_counting_uses_ball_numbers():
    """红球命中计票必须用球号（1-based）对球号，不能用 argsort 索引（0-based）

    构造确定性场景：baseline 预测概率最高的是 1-6 号，下一期开奖恰好是 1-6 号。
    正确计票应得 6 球；0-based 索引 ∩ 球号只能碰上 5 个（1-5）。
    """
    from backtest import walk_forward_backtest

    draws = _synthetic_draws(60, seed=3)
    # 强制每期开奖为固定号码 1-6，baseline 学到的概率分布稳定偏向这些号
    for i in range(60):
        draws[i] = _make_draw([1, 2, 3, 4, 5, 6], (i % 16) + 1, draws[i]["draw_no"])

    results = walk_forward_backtest(draws, start=55, end=57, include_lstm=False)
    for r in results:
        assert r["red_hits"] == 6, f"开奖1-6且预测应全中，实际记 {r['red_hits']} 球: {r}"


def test_robust_recommendation_is_pure_top6():
    """方案A（稳健型）应直接取概率 top6，不做约束过滤

    实证：约束采样 10 种子×150 期全部降低命中（均值-0.17/6, p=0.0001）。
    构造 top6=1-6 号（和值21必被约束拒绝），方案A 仍应原样返回 top6。
    """
    from ssq.models.ensemble import EnsembleModel

    ens = EnsembleModel()
    red_probs = np.full(33, 0.001)
    red_probs[:6] = 0.15
    red_probs /= red_probs.sum()
    blue_probs = np.ones(16) / 16

    for _ in range(5):
        recs = ens.generate_recommendations(red_probs, blue_probs, n_sets=1)
        assert recs[0]["red"] == [1, 2, 3, 4, 5, 6], \
            f"方案A应返回纯top6 [1,2,3,4,5,6]，实际: {recs[0]['red']}"


def test_default_blue_fusion_ignores_xgboost():
    """蓝球默认融合不应含 XGBoost（三份回测一致最差且低于理论随机），红球融合应保留"""
    from ssq.models.ensemble import EnsembleModel

    ens = EnsembleModel()
    xgb_red = np.zeros(33); xgb_red[0] = 1.0   # XGBoost 强推1号红
    xgb_blue = np.zeros(16); xgb_blue[15] = 1.0  # XGBoost 强推16号蓝
    uniform_r = np.ones(33) / 33
    uniform_b = np.ones(16) / 16
    preds_with = {
        "baseline": (uniform_r, uniform_b),
        "xgboost": (xgb_red, xgb_blue),
        "lstm": (uniform_r, uniform_b),
    }
    preds_without = {k: v for k, v in preds_with.items() if k != "xgboost"}

    red_with, blue_with = ens.combine(preds_with)
    red_without, blue_without = ens.combine(preds_without)

    # 蓝球：XGBoost 不参与默认融合 → 其强推信号完全不影响结果
    np.testing.assert_allclose(blue_with, blue_without)
    # 红球：XGBoost 参与 → 1号概率被抬高
    assert red_with[0] > red_without[0]
