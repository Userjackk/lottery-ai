import numpy as np
from ..config import RED_MAX, BLUE_MAX


class EnsembleModel:
    """多模型加权融合"""

    def __init__(self, weights: dict[str, float] = None):
        # 注：降低 XGBoost 权重（对近期频率过拟合），提高 LSTM 和 Baseline 权重
        self.weights = weights or {"baseline": 0.25, "xgboost": 0.45, "lstm": 0.3}
        # 蓝球融合权重：三份独立回测中 XGBoost 蓝球一致最差（3-4%，低于理论6.25%），
        # 剔除后集成蓝球不再被稀释；红球融合保留 XGBoost（单模型最好）
        self.blue_weights = {"baseline": 0.5, "xgboost": 0.0, "lstm": 0.5}

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
        red_total_weight = 0
        blue_total_weight = 0

        for name, (red_p, blue_p) in predictions.items():
            # 归一化每个模型的概率
            red_norm = red_p / (red_p.sum() + 1e-10)
            blue_norm = blue_p / (blue_p.sum() + 1e-10)
            w_red = self.weights.get(name, 0)
            if w_red > 0:
                red_combined += w_red * red_norm
                red_total_weight += w_red
            w_blue = self.blue_weights.get(name, 0)
            if w_blue > 0:
                blue_combined += w_blue * blue_norm
                blue_total_weight += w_blue

        if red_total_weight > 0:
            red_combined /= red_total_weight
        if blue_total_weight > 0:
            blue_combined /= blue_total_weight

        return red_combined, blue_combined

    def generate_recommendations(
        self,
        red_probs: np.ndarray,
        blue_probs: np.ndarray,
        n_sets: int = 3,
        draws: list[dict] = None,
    ) -> list[dict]:
        """
        生成不同风格的推荐号码

        Returns: [{"red": [6个数], "blue": 数, "confidence": 百分比}, ...]
        """
        from ..config import RED_COUNT
        from ..constraints import ConstraintFilter

        cf = ConstraintFilter(draws or [])
        recommendations = []
        sorted_red = np.argsort(red_probs)[::-1]
        sorted_blue = np.argsort(blue_probs)[::-1]

        # 工具函数：按概率加权采样满足约束的红球组合
        def _sample_constrained_red(seed_offset: int = 0) -> list[int] | None:
            rng = np.random.default_rng(seed=(np.random.randint(0, 1_000_000) + seed_offset))
            for _ in range(2000):
                candidates = rng.choice(RED_MAX, size=RED_COUNT, replace=False, p=red_probs / red_probs.sum())
                combo = sorted(int(r + 1) for r in candidates)
                if cf.check(combo):
                    return combo
            #  fallback：从 top 概率里选最接近约束的
            top_pool = sorted_red[:20]
            for _ in range(500):
                pick = rng.choice(top_pool, size=RED_COUNT, replace=False)
                combo = sorted(int(r + 1) for r in pick)
                if cf.check(combo):
                    return combo
            return None

        # 方案A: 稳健型 - 直接取概率 top6（实证：约束采样 10 种子×150期一致降低命中，
        # 均值-0.17/6, p=0.0001，不再做约束过滤）
        red_a = sorted(int(r + 1) for r in sorted_red[:RED_COUNT])
        blue_a = int(sorted_blue[0] + 1)
        conf_a = float(red_probs[sorted_red[:RED_COUNT]].mean() * 100 * RED_COUNT)
        recommendations.append({
            "red": red_a,
            "blue": blue_a,
            "confidence": min(conf_a, 95),
        })

        if n_sets >= 2:
            # 方案B: 均衡型 - top4 + 加权随机2个，约束过滤
            rng = np.random.default_rng(seed=np.random.randint(0, 1_000_000))
            red_b = None
            for _ in range(2000):
                top4 = set(sorted_red[:4])
                remaining = [i for i in range(RED_MAX) if i not in top4]
                rem_probs = red_probs[remaining]
                rem_probs = rem_probs / rem_probs.sum()
                extra2 = rng.choice(remaining, size=2, replace=False, p=rem_probs)
                combo = sorted(list(top4) + list(extra2))
                if cf.check(combo):
                    red_b = [int(r + 1) for r in combo]
                    break
            if red_b is None:
                red_b = _sample_constrained_red(1) or sorted(int(r + 1) for r in sorted_red[:RED_COUNT])
            blue_b_candidates = sorted_blue[:4]
            blue_b = int(np.random.choice(blue_b_candidates)) + 1
            conf_b = float(red_probs[sorted_red[:4]].mean() * 100 * 4 +
                          red_probs[[r - 1 for r in red_b]].mean() * 100 * 2) / RED_COUNT * 1.5
            recommendations.append({
                "red": red_b,
                "blue": blue_b,
                "confidence": min(conf_b, 85),
            })

        if n_sets >= 3:
            # 方案C: 搏冷型 - 侧重遗漏值高且满足约束
            rng = np.random.default_rng(seed=np.random.randint(0, 1_000_000))
            cold_mask = red_probs < np.median(red_probs)
            cold_indices = np.where(cold_mask)[0]
            red_c = None
            if len(cold_indices) >= RED_COUNT:
                cold_probs = red_probs[cold_indices] + 0.01
                cold_probs = cold_probs / cold_probs.sum()
                for _ in range(2000):
                    cold_picks = rng.choice(cold_indices, size=RED_COUNT, replace=False, p=cold_probs)
                    combo = sorted(int(r + 1) for r in cold_picks)
                    if cf.check(combo):
                        red_c = combo
                        break
            if red_c is None:
                red_c = _sample_constrained_red(2) or sorted(int(r + 1) for r in sorted_red[:RED_COUNT])
            blue_c = int(np.random.choice(sorted_blue[:6])) + 1
            conf_c = float(red_probs[[r - 1 for r in red_c]].mean() * 100 * RED_COUNT) * 0.7
            recommendations.append({
                "red": red_c,
                "blue": blue_c,
                "confidence": max(min(conf_c, 70), 30),
            })

        return recommendations
