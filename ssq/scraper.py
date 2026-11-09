import time
import requests
from .config import DATA_URL, DATA_HEADERS, REQUEST_TIMEOUT, RETRY_COUNT, RETRY_DELAY, RED_COUNT, RED_MAX, BLUE_MAX
from .validation import validate_draw


def fetch_page(page_no: int = 1, page_size: int = 2000) -> list[dict]:
    """从福彩官网API抓取开奖数据"""
    params = {
        "name": "ssq",
        "issueCount": page_size,
        "pageNo": page_no,
        "pageSize": page_size,
        "systemType": "PC",
    }
    last_error = None
    for attempt in range(RETRY_COUNT):
        try:
            resp = requests.get(
                DATA_URL, params=params, headers=DATA_HEADERS, timeout=REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
            return _parse_results(data.get("result", []))
        except Exception as e:
            last_error = e
            if attempt < RETRY_COUNT - 1:
                time.sleep(RETRY_DELAY)
    raise RuntimeError(f"抓取数据失败: {last_error}")


def _parse_results(results: list[dict]) -> list[dict]:
    draws = []
    for r in results:
        code = r.get("code", "")
        red_str = r.get("red", "")
        blue_str = r.get("blue", "")
        if not red_str or not blue_str:
            continue
        try:
            reds = sorted(int(x) for x in red_str.split(","))
            blue = int(blue_str)
        except ValueError:
            continue
        if len(reds) != RED_COUNT:
            continue
        if any(not (1 <= x <= RED_MAX) for x in reds) or not (1 <= blue <= BLUE_MAX):
            continue
        draw = {
            "draw_no": code,
            "date": r.get("date", "")[:10],
            "red1": reds[0],
            "red2": reds[1],
            "red3": reds[2],
            "red4": reds[3],
            "red5": reds[4],
            "red6": reds[5],
            "blue": blue,
        }
        if validate_draw(draw):
            draws.append(draw)
    return draws


def fetch_all() -> list[dict]:
    """抓取全部历史数据（单次大请求，API上限~2017期）"""
    draws = fetch_page(1, 5000)
    print(f"  共抓取 {len(draws)} 期")
    return draws


def fetch_latest() -> list[dict]:
    """只抓取最新数据（用于增量更新）"""
    return fetch_page(1, 100)
