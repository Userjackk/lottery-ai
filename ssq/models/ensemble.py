import numpy as np
from ..config import RED_MAX, BLUE_MAX


class EnsembleModel:
    """多模型加权融合"""

    def __init__(self, weights: dict[str, float] = None):
        # 注：降低 XGBoost 权重（对近期频率过拟合），提高 LSTM 和 Baseline 权重
        self.weights = weights or {"baseline": 0.25, "xgboost": 0.45, "lstm": 0.3}

    def combine(
        self,
        predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        融合多个模型的预测概率

        Args:
            predictions: {"model_name": (red_probs_33, blue_probs_16), ...}

        Returns:
            (red_probs, blue_probs) 归一化后的概率
        """
        red_combined = np.zeros(RED_MAX)
        blue_combined = np.zeros(BLUE_MAX)
        total_weight = 0

        for name, (red_p, blue_p) in predictions.items():
            w = self.weights.get(name, 0)
            if w <= 0:
                continue
            # 归一化每个模型的概率
            red_norm = red_p / (red_p.sum() + 1e-10)
            blue_norm = blue_p / (blue_p.sum() + 1e-10)
            red_combined += w * red_norm
            blue_combined += w * blue_norm
            total_weight += w

        if total_weight > 0:
            red_combined /= total_weight
            blue_combined /= total_weight

        return red_combined, blue_combined

    def generate_recommendations(
        self,
        red_probs: np.ndarray,
        blue_probs: np.ndarray,
        n_sets: int = 3,
    ) -> list[dict]:
        """
        生成不同风格的推荐号码

        Returns: [{"red": [6个数], "blue": 数, "confidence": 百分比}, ...]
        """
        from ..config import RED_COUNT

        recommendations = []
        sorted_red = np.argsort(red_probs)[::-1]
        sorted_blue = np.argsort(blue_probs)[::-1]

        # 方案A: 稳健型 - 直接取概率最高的6个红球
        red_a = sorted(int(r + 1) for r in sorted_red[:RED_COUNT])
        blue_a = int(sorted_blue[0] + 1)
        conf_a = float(red_probs[sorted_red[:RED_COUNT]].mean() * 100 * RED_COUNT)
        recommendations.append({
            "red": red_a,
            "blue": blue_a,
            "confidence": min(conf_a, 95),
        })

        if n_sets >= 2:
            # 方案B: 均衡型 - top4 + 加权随机2个
            top4 = set(sorted_red[:4])
            remaining = [i for i in range(RED_MAX) if i not in top4]
            rem_probs = red_probs[remaining]
            rem_probs = rem_probs / rem_probs.sum()
            extra2 = np.random.choice(remaining, size=2, replace=False, p=rem_probs)
            red_b = sorted(list(top4) + list(extra2))
            blue_b_candidates = sorted_blue[:4]
            blue_b = np.random.choice(blue_b_candidates) + 1
            conf_b = float(red_probs[sorted_red[:4]].mean() * 100 * 4 +
                          red_probs[extra2].mean() * 100 * 2) / RED_COUNT * 1.5
            recommendations.append({
                "red": [int(r + 1) for r in red_b],
                "blue": int(blue_b),
                "confidence": min(conf_b, 85),
            })

        if n_sets >= 3:
            # 方案C: 搏冷型 - 侧重遗漏值高的号码
            cold_mask = red_probs < np.median(red_probs)
            cold_indices = np.where(cold_mask)[0]
            if len(cold_indices) >= RED_COUNT:
                cold_probs = red_probs[cold_indices] + 0.01
                cold_probs = cold_probs / cold_probs.sum()
                cold_picks = np.random.choice(cold_indices, size=RED_COUNT, replace=False, p=cold_probs)
            else:
                cold_picks = np.random.choice(RED_MAX, size=RED_COUNT, replace=False, p=red_probs / red_probs.sum())
            red_c = sorted(cold_picks)
            blue_c = np.random.choice(sorted_blue[:6]) + 1
            conf_c = float(red_probs[cold_picks].mean() * 100 * RED_COUNT) * 0.7
            recommendations.append({
                "red": [int(r + 1) for r in red_c],
                "blue": int(blue_c),
                "confidence": max(min(conf_c, 70), 30),
            })

        return recommendations
