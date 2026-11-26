"""双色球奖级数据（销售额 / 各等奖中奖注数与单注奖金）抓取与入库

独立于 ssq/db.py 的 draws 表，单独一张 prize_stats 表。
"""
import time

import requests

from . import db
from .config import DATA_URL, DATA_HEADERS

PAGE_TIMEOUT = 30
RETRY_COUNT = 3
RETRY_DELAY = 2


def _to_int(value) -> int | None:
    """API 里数字字段是字符串，空串/None/非数字统一返回 None"""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def parse_prize_record(raw: dict) -> dict | None:
    """解析单期 API 记录；销售额缺失或为 0 的期次返回 None"""
    sales = _to_int(raw.get("sales"))
    if sales is None or sales <= 0:
        return None

    grades = {g.get("type"): g for g in (raw.get("prizegrades") or [])}
    g1 = grades.get(1) or {}
    g2 = grades.get(2) or {}

    return {
        "draw_no": str(raw.get("code", "")).strip(),
        "sales": sales,
        "poolmoney": _to_int(raw.get("poolmoney")),
        "n1": _to_int(g1.get("typenum")),
        "m1": _to_int(g1.get("typemoney")),
        "n2": _to_int(g2.get("typenum")),
        "m2": _to_int(g2.get("typemoney")),
    }


def fetch_with_prizes(page_size: int = 5000) -> list[dict]:
    """一次抓取全部历史奖级数据（福彩 API 单次上限约 5000 期）"""
    params = {
        "name": "ssq",
        "issueCount": page_size,
        "pageNo": 1,
        "pageSize": page_size,
        "systemType": "PC",
    }
    last_error = None
    for attempt in range(RETRY_COUNT):
        try:
            resp = requests.get(
                DATA_URL, params=params, headers=DATA_HEADERS, timeout=PAGE_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
            rows = [parse_prize_record(r) for r in data.get("result", [])]
            return [r for r in rows if r]
        except Exception as e:
            last_error = e
            if attempt < RETRY_COUNT - 1:
                time.sleep(RETRY_DELAY)
    raise RuntimeError(f"抓取奖级数据失败: {last_error}")


def init_prize_table() -> None:
    with db.get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS prize_stats (
                draw_no   TEXT PRIMARY KEY,
                sales     INTEGER NOT NULL,
                poolmoney INTEGER,
                n1        INTEGER,
                m1        INTEGER,
                n2        INTEGER,
                m2        INTEGER
            );
        """)


def insert_prize_stats(rows: list[dict]) -> int:
    valid = [r for r in rows if r.get("sales") and r["sales"] > 0]
    if not valid:
        return 0
    with db.get_conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO prize_stats "
            "(draw_no, sales, poolmoney, n1, m1, n2, m2) "
            "VALUES (:draw_no, :sales, :poolmoney, :n1, :m1, :n2, :m2)",
            valid,
        )
    return len(valid)


def get_all_prize_stats() -> list[dict]:
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM prize_stats ORDER BY draw_no ASC"
        ).fetchall()
        return [dict(r) for r in rows]