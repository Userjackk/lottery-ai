"""测试数据校验功能"""
import pytest
from ssq.scraper import _parse_results
from ssq.db import insert_draws, init_db, get_all_draws
from ssq import validate_draw


def test_parse_valid_draw():
    """正常福彩 API 数据应被正确解析"""
    raw = [{
        "code": "2026060",
        "date": "2026-09-20T00:00:00+08:00",
        "red": "03,12,15,19,22,30",
        "blue": "08",
    }]
    draws = _parse_results(raw)
    assert len(draws) == 1
    assert draws[0]["draw_no"] == "2026060"
    assert draws[0]["red1"] == 3
    assert draws[0]["red6"] == 30
    assert draws[0]["blue"] == 8


def test_parse_invalid_red_count():
    """红球数量不等于6时应丢弃"""
    raw = [{
        "code": "2026060",
        "date": "2026-09-20",
        "red": "03,12,15,19,22",
        "blue": "08",
    }]
    draws = _parse_results(raw)
    assert len(draws) == 0


def test_parse_invalid_blue():
    """蓝球超出1-16范围时应丢弃"""
    raw = [{
        "code": "2026060",
        "date": "2026-09-20",
        "red": "03,12,15,19,22,30",
        "blue": "20",
    }]
    draws = _parse_results(raw)
    assert len(draws) == 0


def test_validate_draw_function():
    """独立校验函数应能识别重复红球"""
    valid = {
        "draw_no": "2026060",
        "red1": 1, "red2": 2, "red3": 3, "red4": 4, "red5": 5, "red6": 6,
        "blue": 7,
    }
    invalid = {
        "draw_no": "2026060",
        "red1": 1, "red2": 2, "red3": 3, "red4": 4, "red5": 5, "red6": 5,
        "blue": 7,
    }
    assert validate_draw(valid) is True
    assert validate_draw(invalid) is False


def test_insert_draws_skips_invalid(tmp_path, monkeypatch):
    """insert_draws 应跳过非法记录"""
    monkeypatch.setattr("ssq.db.DB_PATH", tmp_path / "ssq.db")
    init_db()
    draws = [
        {"draw_no": "2026060", "date": "2026-09-20", "red1": 1, "red2": 2,
         "red3": 3, "red4": 4, "red5": 5, "red6": 6, "blue": 7},
        {"draw_no": "2026061", "date": "2026-09-22", "red1": 1, "red2": 2,
         "red3": 3, "red4": 4, "red5": 5, "red6": 5, "blue": 7},
    ]
    count = insert_draws(draws)
    assert count == 1
    assert len(get_all_draws()) == 1
