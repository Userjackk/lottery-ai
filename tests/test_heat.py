"""号码热度测量：数据层、热度指数、统计工具"""
from ssq.prize_data import parse_prize_record

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