import numpy as np
from ..config import RED_MAX, BLUE_MAX, RED_COUNT


class BaselineModel:
    """频率加权随机基线模型"""

    def __init__(self):
        self.red_weights = None
        self.blue_weights = None

    def fit(self, draws: list[dict]):
        """基于最近数据计算权重"""
        if not draws:
            return
        window = min(50, len(draws))
        recent = draws[-window:]

        red_count = np.zeros(RED_MAX)
        blue_count = np.zeros(BLUE_MAX)

        for d in recent:
            for key in [f"red{k}" for k in range(1, 7)]:
                red_count[d[key] - 1] += 1
            blue_count[d["blue"] - 1] += 1

        # 遗漏因子：遗漏越久权重越高
        red_omit = np.zeros(RED_MAX)
        blue_omit = np.zeros(BLUE_MAX)
        for i in range(len(draws) - 1, -1, -1):
            d = draws[i]
            for key in [f"red{k}" for k in range(1, 7)]:
                idx = d[key] - 1
                if red_omit[idx] == 0:
                    red_omit[idx] = len(draws) - 1 - i
            if blue_omit[d["blue"] - 1] == 0:
                blue_omit[d["blue"] - 1] = len(draws) - 1 - i

        # 权重 = 频率 * (1 + 遗漏因子)
        # 注：提高遗漏因子，让长期未出的号码获得更多权重
        omit_factor = 0.05
        self.red_weights = red_count * (1 + red_omit * omit_factor)
        self.blue_weights = blue_count * (1 + blue_omit * omit_factor)

        # 归一化
        self.red_weights = self.red_weights / self.red_weights.sum()
        self.blue_weights = self.blue_weights / self.blue_weights.sum()

    def predict_proba(self) -> tuple[np.ndarray, np.ndarray]:
        """返回红球33维概率向量和蓝球16维概率向量"""
        return self.red_weights.copy(), self.blue_weights.copy()
