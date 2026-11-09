import numpy as np
from .config import RED_MAX, BLUE_MAX, RED_COUNT


def _reds(draw: dict) -> list[int]:
    return [draw["red1"], draw["red2"], draw["red3"], draw["red4"], draw["red5"], draw["red6"]]


def build_feature_matrix(draws: list[dict], lookback_windows: list[int] = None):
    """
    构建特征矩阵和标签（向量化版本）

    Returns:
        X: (n_samples, n_features) 特征矩阵
        y_red: (n_samples, 33) 红球标签（二值向量）
        y_blue: (n_samples, 16) 蓝球标签（二值向量）
    """
    if lookback_windows is None:
        lookback_windows = [20, 30, 50]

    min_history = max(lookback_windows) + 1
    if len(draws) < min_history:
        raise ValueError(f"数据不足，至少需要{min_history}期")

    # 预构建 one-hot 矩阵（一次循环）
    n = len(draws)
    red_matrix = np.zeros((n, RED_MAX), dtype=np.float32)
    blue_vec = np.zeros(n, dtype=np.int32)
    for idx, d in enumerate(draws):
        for key in [f"red{i}" for i in range(1, 7)]:
            red_matrix[idx, d[key] - 1] = 1.0
        blue_vec[idx] = d["blue"]

    # 窗口频率用累加和，O(1) 切片
    red_cumsum = np.cumsum(red_matrix, axis=0)
    blue_onehot = np.zeros((n, BLUE_MAX), dtype=np.float32)
    for b in range(BLUE_MAX):
        blue_onehot[:, b] = (blue_vec == b + 1).astype(np.float32)
    blue_cumsum = np.cumsum(blue_onehot, axis=0)

    X_list, y_red_list, y_blue_list = [], [], []

    # 标签用下一期号码：用第i期特征预测第i+1期开奖
    for i in range(min_history, n - 1):
        features = _extract_vectorized(
            i, draws, red_matrix, blue_vec, red_cumsum, blue_cumsum, lookback_windows
        )
        red_label = red_matrix[i + 1].copy()
        blue_label = np.zeros(BLUE_MAX, dtype=np.float32)
        blue_label[draws[i + 1]["blue"] - 1] = 1.0

        X_list.append(features)
        y_red_list.append(red_label)
        y_blue_list.append(blue_label)

    return np.array(X_list, dtype=np.float32), np.array(y_red_list), np.array(y_blue_list)


def build_next_features(draws: list[dict], lookback_windows: list[int] = None) -> np.ndarray:
    """为下一期预测构建特征向量"""
    if lookback_windows is None:
        lookback_windows = [20, 30, 50]

    n = len(draws)
    red_matrix = np.zeros((n, RED_MAX), dtype=np.float32)
    blue_vec = np.zeros(n, dtype=np.int32)
    for idx, d in enumerate(draws):
        for key in [f"red{i}" for i in range(1, 7)]:
            red_matrix[idx, d[key] - 1] = 1.0
        blue_vec[idx] = d["blue"]

    red_cumsum = np.cumsum(red_matrix, axis=0)
    blue_onehot = np.zeros((n, BLUE_MAX), dtype=np.float32)
    for b in range(BLUE_MAX):
        blue_onehot[:, b] = (blue_vec == b + 1).astype(np.float32)
    blue_cumsum = np.cumsum(blue_onehot, axis=0)

    features = _extract_vectorized(n - 1, draws, red_matrix, blue_vec, red_cumsum, blue_cumsum, lookback_windows)
    return features.reshape(1, -1)


def _extract_vectorized(
    pos: int,
    draws: list[dict],
    red_matrix: np.ndarray,
    blue_vec: np.ndarray,
    red_cumsum: np.ndarray,
    blue_cumsum: np.ndarray,
    windows: list[int],
) -> np.ndarray:
    """向量化提取单期特征

    特征设计原则：
    - 遗漏值为主（反映号码多久没出，信号更稳定）
    - 长窗口频率为主（30/50期），短窗口弱化（避免近因效应）
    - 增加遗漏比率（遗漏值/期望间隔），消除号码自身频率差异
    """
    n = pos + 1  # 当前可见数据量
    features = []

    # === 红球遗漏值（逆向扫描，提前终止）===
    red_omits = np.full(RED_MAX, n, dtype=np.float32)
    for j in range(pos, -1, -1):
        appeared = np.where(red_matrix[j] == 1)[0]
        for num in appeared:
            if red_omits[num] == n:
                red_omits[num] = pos - j
        if (red_omits < n).all():
            break

    # === 蓝球遗漏值 ===
    blue_omits = np.full(BLUE_MAX, n, dtype=np.float32)
    for j in range(pos, -1, -1):
        b = blue_vec[j] - 1
        if blue_omits[b] == n:
            blue_omits[b] = pos - j
        if (blue_omits < n).all():
            break

    # 期望间隔：红球平均 33/6≈5.5 期出一次，蓝球 16 期出一次
    red_expected_gap = RED_MAX / RED_COUNT  # 5.5
    blue_expected_gap = BLUE_MAX  # 16

    # === 红球特征：遗漏 + 遗漏比率 + 长窗口频率 ===
    for num in range(RED_MAX):
        # 遗漏值（归一化）
        features.append(red_omits[num] / n)
        # 遗漏比率：遗漏值/期望间隔，>1 表示超过平均间隔未出
        features.append(red_omits[num] / red_expected_gap)
        # 窗口频率（跳过10期窗口，用20/30/50）
        for w in windows:
            if w < 15:
                continue  # 跳过10期窗口
            end = pos + 1
            start = max(0, end - w)
            s = red_cumsum[end - 1, num] - (red_cumsum[start - 1, num] if start > 0 else 0)
            features.append(float(s / (end - start)))

    # === 蓝球特征 ===
    for num in range(BLUE_MAX):
        features.append(blue_omits[num] / n)
        features.append(blue_omits[num] / blue_expected_gap)
        for w in windows:
            if w < 15:
                continue
            end = pos + 1
            start = max(0, end - w)
            s = blue_cumsum[end - 1, num] - (blue_cumsum[start - 1, num] if start > 0 else 0)
            features.append(float(s / (end - start)))

    # === 上一期全局特征 ===
    prev = _reds(draws[pos])
    features.append(sum(1 for r in prev if r % 2 == 1) / RED_COUNT)
    features.append(sum(1 for r in prev if r > 16) / RED_COUNT)
    features.append(sum(1 for r in prev if r <= 11) / RED_COUNT)
    features.append(sum(1 for r in prev if 12 <= r <= 22) / RED_COUNT)
    features.append(sum(1 for r in prev if r >= 23) / RED_COUNT)
    features.append(sum(prev) / (RED_MAX * RED_COUNT))
    consecutive = sum(1 for i in range(len(prev) - 1) if prev[i + 1] - prev[i] == 1)
    features.append(consecutive / RED_COUNT)

    # === 相邻期差值特征 ===
    if pos >= 1:
        prev2 = _reds(draws[pos - 1])
        diffs = [abs(a - b) for a, b in zip(sorted(prev), sorted(prev2))]
        features.append(np.mean(diffs) / RED_MAX)
        features.append(np.max(diffs) / RED_MAX)
    else:
        features.append(0.0)
        features.append(0.0)

    # === 冷热号比例 ===
    hot_count = float(np.sum(red_omits == 0))
    theoretical_avg = n * RED_COUNT / RED_MAX
    long_omit_count = float(np.sum(red_omits > theoretical_avg))
    features.append(hot_count / RED_MAX)
    features.append(long_omit_count / RED_MAX)

    # === 蓝球/红球遗漏方差 ===
    features.append(float(np.std(blue_omits) / n))
    features.append(float(np.std(red_omits) / n))

    # === 扩展全局特征：跨度 + AC值 ===
    prev_sorted = sorted(prev)
    span = prev_sorted[-1] - prev_sorted[0]
    features.append(span / RED_MAX)

    ac_value = _ac_value(prev_sorted)
    features.append(ac_value / (RED_MAX - RED_COUNT))  # 归一化到 [0,1] 左右

    return np.array(features, dtype=np.float32)


def _ac_value(reds: list[int]) -> int:
    """计算 AC 值（Arithmetic Complexity）：任意两数差的种类数 - 5

    双色球标准 AC 值 = 不同差值个数 - 5，范围 0 ~ 27。
    """
    from itertools import combinations
    diffs = {b - a for a, b in combinations(reds, 2)}
    return len(diffs) - (RED_COUNT - 1)
