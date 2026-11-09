"""开奖组合统计约束过滤"""
import numpy as np
from .config import RED_MAX, RED_COUNT


def _reds(d):
    return [d["red1"], d["red2"], d["red3"], d["red4"], d["red5"], d["red6"]]


class ConstraintFilter:
    """基于历史统计约束过滤号码组合"""

    def __init__(self, draws: list[dict]):
        self.draws = draws

    def check(self, reds: list[int]) -> bool:
        """检查组合是否符合常见统计特征"""
        reds = sorted(reds)
        if len(set(reds)) != RED_COUNT:
            return False
        if any(not (1 <= r <= RED_MAX) for r in reds):
            return False

        s = sum(reds)
        if s < 70 or s > 130:
            return False

        odd = sum(1 for r in reds if r % 2 == 1)
        if odd < 2 or odd > 4:
            return False

        z1 = sum(1 for r in reds if r <= 11)
        z2 = sum(1 for r in reds if 12 <= r <= 22)
        z3 = sum(1 for r in reds if r >= 23)
        if (z1, z2, z3) not in {
            (2, 2, 2), (3, 1, 2), (1, 3, 2), (3, 2, 1),
            (1, 2, 3), (2, 3, 1), (2, 1, 3), (1, 4, 1), (4, 1, 1),
        }:
            return False

        span = reds[-1] - reds[0]
        if span < 15 or span > 32:
            return False

        consecutive = sum(1 for i in range(len(reds) - 1) if reds[i + 1] - reds[i] == 1)
        if consecutive > 2:
            return False

        return True
