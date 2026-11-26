"""号码热度测量：数据层、热度指数、统计工具"""
import numpy as np

from ssq.prize_data import parse_prize_record
from ssq.heat import (
    FEATURE_KEYS, P1, P2, bh_correct, bootstrap_ci, build_dataset,
    check_heat_uniformity, combo_features, has_arithmetic_pattern,
    heat_index, ols_hc1, quantile_groups,
)
from heat_analysis import format_feature_table, format_report

_SAMPLE = {
    "code": "2026135", "date": "2026-11-24(二)", "red": "02,10,16,22,26,30", "blue": "08",
    "sales": "409302648", "poolmoney": "1148231435",
    "prizegrades": [
        {"type": 1, "typenum": "13", "typemoney": "6629354"},
        {"type": 2, "typenum": "188", "typemoney": "140835"},
        {"type": 3, "typenum": "1708", "typemoney": "3000"},
        {"type": 7, "typenum": "", "typemoney": ""},
    ],
}


def test_parse_prize_record_extracts_fields():
    row = parse_prize_record(_SAMPLE)
    assert row["draw_no"] == "2026135"
    assert row["sales"] == 409302648
    assert row["poolmoney"] == 1148231435
    assert row["n1"] == 13 and row["m1"] == 6629354
    assert row["n2"] == 188 and row["m2"] == 140835


def test_parse_prize_record_skips_missing_sales():
    raw = dict(_SAMPLE, sales="")
    assert parse_prize_record(raw) is None


def test_parse_prize_record_zero_sales_returns_none():
    raw = dict(_SAMPLE, sales="0")
    assert parse_prize_record(raw) is None


def test_parse_prize_record_handles_blank_grade_numbers():
    """type 7 恒为空字符串，不得抛异常；缺失的等级记为 None"""
    raw = dict(_SAMPLE, prizegrades=[{"type": 1, "typenum": "", "typemoney": ""}])
    row = parse_prize_record(raw)
    assert row["n1"] is None and row["m1"] is None
    assert row["n2"] is None


def test_p1_p2_match_lottery_odds():
    assert abs(P1 - 1 / 17_721_088) < 1e-15
    assert abs(P2 - 15 / 17_721_088) < 1e-15


def test_heat_index_uniform_is_one():
    """销量正好覆盖 1/P2 注时，1 注中奖对应 H2=1"""
    sales = int(2 / P2)
    assert abs(heat_index(1, sales, P2) - 1.0) < 1e-3


def test_heat_index_halved_sales_doubles():
    """同样 1 注中奖，销量减半 → 热度翻倍"""
    sales = int(2 / P2)
    assert abs(heat_index(1, sales // 2, P2) - 2.0) < 1e-2


def test_heat_index_zero_sales_is_nan():
    assert np.isnan(heat_index(5, 0, P2))


def test_heat_index_missing_prize_count_is_nan():
    assert np.isnan(heat_index(None, 400_000_000, P2))


def test_has_arithmetic_pattern():
    assert has_arithmetic_pattern([1, 2, 3, 20, 25, 30])
    assert has_arithmetic_pattern([1, 3, 5, 20, 22, 33])
    assert not has_arithmetic_pattern([2, 7, 13, 21, 28, 32])


def test_combo_features_known_combo():
    f = combo_features([1, 2, 3, 12, 31, 33], blue=5)
    assert f["small_count"] == 4.0       # 1,2,3,12
    assert f["date_count"] == 5.0        # 除 33 外都在 1-31
    assert f["sum_value"] == 82.0
    assert f["span"] == 32.0
    assert f["longest_run"] == 3.0
    assert f["consec_pairs"] == 2.0      # (1,2) (2,3)
    assert f["same_tail"] == 3.0         # 1/31, 2/12, 3/33
    assert f["odd_count"] == 4.0
    assert abs(f["zone_balance"] - float(np.std([3, 1, 2]))) < 1e-9  # 1-11:3, 12-22:1, 23-33:2
    assert f["has_arith3"] == 1.0
    assert f["blue_small"] == 1.0
    assert f["repeat_prev"] == 0.0


def test_combo_features_repeat_prev_counts_overlap():
    f = combo_features([1, 2, 3, 12, 31, 33], blue=14, prev_reds=[3, 33, 7, 8, 9, 10])
    assert f["repeat_prev"] == 2.0
    assert f["blue_small"] == 0.0


def test_combo_features_keys_match_constant():
    f = combo_features([2, 10, 16, 22, 26, 30], blue=8)
    assert set(f.keys()) == set(FEATURE_KEYS)


def test_ols_recovers_known_coefficient():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 1))
    y = 3.0 + 2.0 * x[:, 0] + rng.normal(scale=0.1, size=500)
    fit = ols_hc1(x, y)
    assert abs(fit["beta"][0] - 3.0) < 0.05
    assert abs(fit["beta"][1] - 2.0) < 0.05
    assert fit["p"][1] < 1e-10


def test_ols_no_effect_has_large_p():
    """独立随机 x/y：5 个种子中至少 4 个不显著（单种子可能偶然显著）"""
    large = 0
    for seed in range(5):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(500, 1))
        y = rng.normal(size=500)
        if ols_hc1(x, y)["p"][1] > 0.05:
            large += 1
    assert large >= 4


def test_ols_zero_variance_column_returns_nan():
    """X 含零方差列时不得抛异常，该列输出为 nan"""
    rng = np.random.default_rng(2)
    x = np.column_stack([rng.normal(size=50), np.ones(50)])
    y = rng.normal(size=50)
    fit = ols_hc1(x, y)
    assert np.isnan(fit["se"][2])
    assert np.isnan(fit["p"][2])
    assert fit["beta"][2] == 0.0


def test_bh_correct_simple_values():
    adj = bh_correct([0.01, 0.02, 0.03, 0.04])
    assert np.allclose(adj, [0.04, 0.04, 0.04, 0.04])


def test_bh_correct_monotone_and_bounded():
    rng = np.random.default_rng(3)
    p = rng.uniform(size=40)
    adj = bh_correct(p.tolist())
    assert (adj >= p - 1e-12).all()
    assert (adj <= 1.0 + 1e-12).all()


def test_bh_correct_passes_nan_through():
    adj = bh_correct([0.01, float("nan"), 0.04])
    assert np.isnan(adj[1])
    assert not np.isnan(adj[0]) and not np.isnan(adj[2])


def test_quantile_groups_three_balanced():
    g = quantile_groups(np.arange(300.0), n_groups=3)
    assert set(g) == {0, 1, 2}
    assert abs((g == 0).sum() - 100) <= 1
    assert abs((g == 2).sum() - 100) <= 1


def test_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(4)
    v = rng.normal(loc=1.5, size=200)
    lo, hi = bootstrap_ci(v, n_boot=1000, seed=42)
    assert lo < v.mean() < hi


def _prize(draw_no, sales, n1, n2):
    return {"draw_no": draw_no, "sales": sales, "poolmoney": 0,
            "n1": n1, "m1": 0, "n2": n2, "m2": 0}


def _draw(draw_no, reds, blue):
    d = {"draw_no": draw_no, "date": "2026-01-01", "blue": blue}
    for i, r in enumerate(reds, 1):
        d[f"red{i}"] = r
    return d


def test_build_dataset_merges_and_computes_heat():
    sales = int(2 / P2)
    rows = build_dataset(
        [_prize("2026001", sales, 1, 1)],
        [_draw("2026001", [2, 10, 16, 22, 26, 30], 8)],
    )
    assert len(rows) == 1
    r = rows[0]
    assert abs(r["h2"] - 1.0) < 1e-3
    assert abs(r["h1"] - P2 / P1) < 1e-3
    assert r["small_count"] == 2.0
    assert r["blue_small"] == 1.0


def test_build_dataset_skips_unmatched_draw():
    """奖级数据缺该期时跳过，不静默塞 0"""
    rows = build_dataset(
        [_prize("2026002", 400_000_000, 5, 100)],
        [_draw("2026001", [1, 2, 3, 4, 5, 6], 1)],
    )
    assert rows == []


def test_build_dataset_sets_repeat_prev_from_previous_draw():
    sales = int(2 / P2)
    rows = build_dataset(
        [_prize("2026001", sales, 1, 1), _prize("2026002", sales, 1, 1)],
        [_draw("2026001", [1, 2, 3, 4, 5, 6], 1),
         _draw("2026002", [6, 7, 8, 9, 10, 11], 2)],
    )
    assert rows[0]["repeat_prev"] == 0.0
    assert rows[1]["repeat_prev"] == 1.0


def test_check_heat_uniformity_ok_when_mean_near_one():
    res = check_heat_uniformity(np.array([0.9, 1.0, 1.1, 1.05]))
    assert res["ok"] is True
    assert res["n"] == 4


def test_check_heat_uniformity_fails_when_mean_far():
    res = check_heat_uniformity(np.array([2.0, 2.1, 1.9]))
    assert res["ok"] is False


def test_check_heat_uniformity_ignores_nan():
    res = check_heat_uniformity(np.array([np.nan, 1.0, 1.0]))
    assert res["n"] == 2
    assert res["ok"] is True


def _synthetic_rows(n=300, effect=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        small = float(rng.integers(0, 7))
        row = {k: 0.0 for k in FEATURE_KEYS}
        row.update({
            "draw_no": f"2026{i:03d}", "sales": 400_000_000,
            "n1": 5, "n2": 150, "h1": 1.0,
            "h2": float(np.exp(effect * (small - 3) + rng.normal(scale=0.3))),
            "small_count": small, "date_count": 5.0, "blue_small": 1.0,
        })
        rows.append(row)
    return rows


def test_format_feature_table_has_all_features_and_direction():
    rows = _synthetic_rows(n=300, effect=0.4)
    text = format_feature_table(rows, FEATURE_KEYS)
    for k in FEATURE_KEYS:
        assert k in text
    assert "small_count" in text
    # 效应方向要能看见符号
    assert "+" in text
    # 只有 small_count 有真实效应，它必须被判为显著(正)，否则是列下标错位
    line = next(l for l in text.splitlines() if l.startswith("| small_count"))
    assert "显著(正)" in line


def test_format_feature_table_marks_insignificant():
    rows = _synthetic_rows(n=300, effect=0.0)
    text = format_feature_table(rows, FEATURE_KEYS)
    assert "不显著" in text


def test_format_report_contains_uniformity_and_top_draws():
    rows = _synthetic_rows(n=60, effect=0.2)
    uniformity = {"mean": 1.02, "n": 60, "ok": True}
    text = format_report(rows, uniformity, "TABLE", top_n=5)
    assert "1.02" in text
    assert "TABLE" in text
    assert "2026" in text


def test_feature_table_is_gbk_encodable():
    """Windows 控制台是 GBK，表格里不得出现 U+2212 之类无法编码的字符"""
    rows = _synthetic_rows(n=300, effect=0.4)
    text = format_feature_table(rows, FEATURE_KEYS)
    text.encode("gbk")


def test_feature_table_handles_skewed_binary_feature():
    """极度偏斜的二元特征会出现空分档，不得输出 nan 或抛异常"""
    rows = _synthetic_rows(n=300, effect=0.3)
    for i, r in enumerate(rows):
        r["has_arith3"] = 1.0 if i < 8 else 0.0
    text = format_feature_table(rows, FEATURE_KEYS)
    line = next(l for l in text.splitlines() if l.startswith("| has_arith3"))
    assert "nan" not in line.lower()
    assert "—" in line


def test_feature_table_has_crosscheck_section():
    """共线时单变量与多变量可能给出相反符号，必须报出来"""
    rows = _synthetic_rows(n=300, effect=0.4)
    text = format_feature_table(rows, FEATURE_KEYS)
    assert "交叉核对" in text
    cross = text.split("### 交叉核对")[1]
    line = next(l for l in cross.splitlines() if l.startswith("| small_count |"))
    assert "一致" in line