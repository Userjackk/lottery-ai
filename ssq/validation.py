"""开奖数据校验"""
from .config import RED_MAX, BLUE_MAX, RED_COUNT


def validate_draw(draw: dict) -> bool:
    """校验单期开奖记录是否合法

    检查项：
    - 期号非空
    - 红球6个，范围1-33，不重复
    - 蓝球1个，范围1-16
    """
    if not draw.get("draw_no"):
        return False

    reds = [draw.get(f"red{i}") for i in range(1, RED_COUNT + 1)]
    if any(r is None for r in reds):
        return False
    try:
        reds = [int(r) for r in reds]
    except (ValueError, TypeError):
        return False

    if len(set(reds)) != RED_COUNT:
        return False
    if any(not (1 <= r <= RED_MAX) for r in reds):
        return False

    blue = draw.get("blue")
    if blue is None:
        return False
    try:
        blue = int(blue)
    except (ValueError, TypeError):
        return False
    if not (1 <= blue <= BLUE_MAX):
        return False

    return True
