"""号码热度测量：数据层、热度指数、统计工具"""
import numpy as np

from ssq.prize_data import parse_prize_record
from ssq.heat import (
    FEATURE_KEYS, P1, P2, combo_features, has_arithmetic_pattern, heat_index,
)

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