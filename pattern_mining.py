"""规律挖掘：从历史数据中寻找非随机的可利用模式"""
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from ssq.db import init_db, get_all_draws
from ssq.config import RED_MAX, BLUE_MAX, RED_COUNT


def _reds(d):
    return [d["red1"], d["red2"], d["red3"], d["red4"], d["red5"], d["red6"]]


# ============================================================
# 1. 号码对共现分析（关联规则）
# ============================================================
def pair_cooccurrence(draws, window=200):
    """哪些号码对倾向于一起出现"""
    recent = draws[-window:]
    pair_count = Counter()
    for d in recent:
        reds = _reds(d)
        for a, b in combinations(sorted(reds), 2):
            pair_count[(a, b)] += 1

    # 理论期望：每对在 6/33 * 5/32 * N 期中出现
    expected = 6 * 5 / (33 * 32) * len(recent)
    print(f"\n{'='*50}")
    print(f"【号码对共现分析】窗口={window}期, 理论期望={expected:.1f}次")
    print(f"{'='*50}")

    top_pairs = pair_count.most_common(15)
    print(f"\n高频共现 TOP15:")
    for (a, b), cnt in top_pairs:
        ratio = cnt / expected
        print(f"  ({a:02d},{b:02d}): {cnt}次, {ratio:.1f}倍期望")

    # 显著低于期望的对（互斥倾向）
    all_pairs = set(combinations(range(1, RED_MAX + 1), 2))
    low_pairs = [(p, pair_count.get(p, 0)) for p in all_pairs]
    low_pairs.sort(key=lambda x: x[1])
    print(f"\n低频共现 TOP10（互斥倾向）:")
    for (a, b), cnt in low_pairs[:10]:
        ratio = cnt / expected if expected > 0 else 0
        print(f"  ({a:02d},{b:02d}): {cnt}次, {ratio:.1f}倍期望")

    return pair_count


# ============================================================
# 2. 马尔可夫链（转移概率）
# ============================================================
def markov_analysis(draws, window=500):
    """前一期出现的号码对下一期各号码概率的影响"""
    recent = draws[-window:]

    # 统计：号码 i 出现后，号码 j 在下一期出现的次数
    trans_count = np.zeros((RED_MAX, RED_MAX))  # [prev][next]
    trans_total = np.zeros(RED_MAX)  # 号码 i 出现了多少次

    for idx in range(len(recent) - 1):
        curr_reds = _reds(recent[idx])
        next_reds = set(_reds(recent[idx + 1]))
        for r in curr_reds:
            i = r - 1
            trans_total[i] += 1
            for nr in next_reds:
                j = nr - 1
                trans_count[i][j] += 1

    # 转移概率矩阵
    trans_prob = np.zeros_like(trans_count)
    for i in range(RED_MAX):
        if trans_total[i] > 0:
            trans_prob[i] = trans_count[i] / trans_total[i]

    # 基准概率：每号出现概率 = 6/33
    base_prob = 6 / RED_MAX

    print(f"\n{'='*50}")
    print(f"【马尔可夫链分析】窗口={window}期")
    print(f"{'='*50}")
    print(f"基准概率: {base_prob:.3f}")

    # 找最强转移关系
    print(f"\n最强正向转移（i出现后j更易出现）:")
    transfers = []
    for i in range(RED_MAX):
        for j in range(RED_MAX):
            if i != j and trans_total[i] >= 20:
                ratio = trans_prob[i][j] / base_prob
                transfers.append((i+1, j+1, trans_prob[i][j], ratio, int(trans_total[i])))
    transfers.sort(key=lambda x: x[3], reverse=True)
    for a, b, prob, ratio, cnt in transfers[:15]:
        print(f"  {a:02d}→{b:02d}: {prob:.3f} ({ratio:.2f}x基准, 样本{cnt})")

    print(f"\n最强负向转移（i出现后j更难出现）:")
    transfers.sort(key=lambda x: x[3])
    for a, b, prob, ratio, cnt in transfers[:10]:
        print(f"  {a:02d}→{b:02d}: {prob:.3f} ({ratio:.2f}x基准, 样本{cnt})")

    return trans_prob


# ============================================================
# 3. 号码周期性分析
# ============================================================
def periodicity_analysis(draws, max_period=50):
    """号码出现是否存在周期性"""
    n = len(draws)
    print(f"\n{'='*50}")
    print(f"【周期性分析】")
    print(f"{'='*50}")

    # 为每个号码计算出现间隔
    best_periods = []
    for num in range(1, RED_MAX + 1):
        # 该号码出现的位置
        positions = []
        for idx, d in enumerate(draws):
            if num in _reds(d):
                positions.append(idx)

        if len(positions) < 5:
            continue

        # 计算间隔序列
        intervals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)

        # 间隔的自相关（找周期性）
        if len(intervals) >= 10:
            # 用最近的间隔序列
            recent_int = intervals[-50:]
            autocorr = []
            for lag in range(1, min(max_period, len(recent_int) // 2)):
                if len(recent_int) > lag:
                    c = np.corrcoef(recent_int[:-lag], recent_int[lag:])[0, 1]
                    autocorr.append((lag, c))

            if autocorr:
                best_lag, best_corr = max(autocorr, key=lambda x: abs(x[1]))
                best_periods.append((num, avg_interval, std_interval, best_lag, best_corr))

    # 按自相关强度排序
    best_periods.sort(key=lambda x: abs(x[4]), reverse=True)

    print(f"\n周期性最强的号码 TOP10:")
    print(f"{'号码':>4} {'平均间隔':>8} {'间隔标准差':>10} {'最佳周期':>8} {'自相关':>8}")
    for num, avg_i, std_i, lag, corr in best_periods[:10]:
        print(f"  {num:02d}   {avg_i:6.1f}期   {std_i:6.1f}     {lag:3d}期    {corr:+.3f}")

    # 预测：哪些号码当前接近其周期峰值
    print(f"\n当前处于周期峰值的号码（遗漏值≈周期）:")
    for d_last_idx in [len(draws) - 1]:
        for num, avg_i, std_i, lag, corr in best_periods:
            if abs(corr) < 0.15:
                continue
            # 当前遗漏值
            omit = 0
            for idx in range(d_last_idx, -1, -1):
                if num in _reds(draws[idx]):
                    omit = d_last_idx - idx
                    break
            # 如果遗漏值接近周期，该号码可能即将出现
            if lag > 0 and 0.7 * lag <= omit <= 1.3 * lag:
                print(f"  号码{num:02d}: 遗漏{omit}期, 周期{lag}期, 自相关{corr:+.3f}")

    return best_periods


# ============================================================
# 4. 间距模式分析
# ============================================================
def gap_pattern_analysis(draws, window=200):
    """红球之间的间距（差值）是否有规律"""
    recent = draws[-window:]
    print(f"\n{'='*50}")
    print(f"【间距模式分析】窗口={window}期")
    print(f"{'='*50}")

    # 统计相邻红球间距
    gap_counter = Counter()
    for d in recent:
        reds = sorted(_reds(d))
        for i in range(len(reds) - 1):
            gap = reds[i+1] - reds[i]
            gap_counter[gap] += 1

    total_gaps = sum(gap_counter.values())
    print(f"\n间距分布（共{total_gaps}个间距）:")
    for gap in range(1, 20):
        cnt = gap_counter.get(gap, 0)
        pct = cnt / total_gaps * 100
        bar = "█" * int(pct)
        if pct > 3:
            print(f"  间距{gap:2d}: {cnt:4d}次 ({pct:5.1f}%) {bar}")

    # 间距序列的自相关
    gap_sequences = []
    for d in recent:
        reds = sorted(_reds(d))
        gaps = [reds[i+1] - reds[i] for i in range(5)]
        gap_sequences.append(gaps)

    # 各位置间距的均值和标准差
    print(f"\n各位置间距统计:")
    for pos in range(5):
        gaps_at_pos = [gs[pos] for gs in gap_sequences]
        print(f"  位置{pos+1}: 均值={np.mean(gaps_at_pos):.2f}, 标准差={np.std(gaps_at_pos):.2f}")

    return gap_counter


# ============================================================
# 5. 区间轮动分析
# ============================================================
def zone_rotation_analysis(draws, window=100):
    """三个区间的号码是否呈现轮动规律"""
    recent = draws[-window:]
    print(f"\n{'='*50}")
    print(f"【区间轮动分析】窗口={window}期")
    print(f"{'='*50}")

    # 区间定义：1-11, 12-22, 23-33
    zone_patterns = []
    for d in recent:
        reds = _reds(d)
        z1 = sum(1 for r in reds if r <= 11)
        z2 = sum(1 for r in reds if 12 <= r <= 22)
        z3 = sum(1 for r in reds if r >= 23)
        zone_patterns.append((z1, z2, z3))

    # 区间分布统计
    pattern_count = Counter(zone_patterns)
    print(f"\n区间分布 TOP10:")
    for (z1, z2, z3), cnt in pattern_count.most_common(10):
        print(f"  [{z1},{z2},{z3}]: {cnt}次 ({cnt/window*100:.0f}%)")

    # 区间转移：连续两期的区间变化
    print(f"\n区间转移模式（前→后）:")
    trans = Counter()
    for i in range(len(zone_patterns) - 1):
        trans[(zone_patterns[i], zone_patterns[i+1])] += 1

    top_trans = trans.most_common(10)
    for (prev, curr), cnt in top_trans:
        print(f"  {prev}→{curr}: {cnt}次")

    return zone_patterns


# ============================================================
# 6. 蓝球规律分析
# ============================================================
def blue_pattern_analysis(draws, window=200):
    """蓝球是否有可利用的规律"""
    recent = draws[-window:]
    print(f"\n{'='*50}")
    print(f"【蓝球规律分析】窗口={window}期")
    print(f"{'='*50}")

    blues = [d["blue"] for d in recent]

    # 蓝球与上期红球的关系
    print(f"\n蓝球与上期红球的关系:")
    blue_from_red = Counter()  # 蓝球是否在上期红球中出现
    for i in range(1, len(recent)):
        prev_reds = set(_reds(recent[i-1]))
        curr_blue = recent[i]["blue"]
        blue_from_red[curr_blue in prev_reds] += 1
    print(f"  蓝球在上期红球中: {blue_from_red[True]}次 ({blue_from_red[True]/(len(recent)-1)*100:.1f}%)")
    print(f"  蓝球不在上期红球中: {blue_from_red[False]}次 ({blue_from_red[False]/(len(recent)-1)*100:.1f}%)")
    expected_in = 6 / 33 * 100
    print(f"  理论期望: {expected_in:.1f}%")

    # 蓝球连续出现相同号码的概率
    consecutive_same = sum(1 for i in range(1, len(blues)) if blues[i] == blues[i-1])
    print(f"\n连续相同蓝球: {consecutive_same}次 ({consecutive_same/(len(blues)-1)*100:.1f}%)")
    print(f"  理论期望: {1/BLUE_MAX*100:.1f}%")

    # 蓝球与上期蓝球的差值
    blue_diffs = [blues[i] - blues[i-1] for i in range(1, len(blues))]
    print(f"\n蓝球差值分布:")
    diff_counter = Counter(blue_diffs)
    for diff in range(-10, 11):
        cnt = diff_counter.get(diff, 0)
        if cnt > 5:
            print(f"  差{diff:+d}: {cnt}次 ({cnt/len(blue_diffs)*100:.1f}%)")

    # 蓝球遗漏值当前状态
    blue_omit = {}
    for b in range(1, BLUE_MAX + 1):
        for idx in range(len(recent) - 1, -1, -1):
            if recent[idx]["blue"] == b:
                blue_omit[b] = len(recent) - 1 - idx
                break
        else:
            blue_omit[b] = len(recent)

    avg_omit = len(recent) / BLUE_MAX
    print(f"\n当前蓝球遗漏值（平均={avg_omit:.0f}期）:")
    for b in sorted(blue_omit, key=blue_omit.get, reverse=True):
        omit = blue_omit[b]
        status = "★高遗漏" if omit > avg_omit * 1.5 else ""
        if status or omit < 3:
            print(f"  蓝球{b:02d}: 遗漏{omit}期 {status}")

    return blue_omit


# ============================================================
# 7. 综合规律预测器
# ============================================================
class PatternPredictor:
    """基于挖掘到的规律进行预测"""

    def __init__(self, draws):
        self.draws = draws
        self.n = len(draws)

    def predict(self, n_sets=3):
        """基于多种规律模式生成预测"""
        scores_red = np.zeros(RED_MAX)
        scores_blue = np.zeros(BLUE_MAX)

        # === 规律1: 马尔可夫转移（最强信号，加大权重）===
        last_reds = _reds(self.draws[-1])
        trans_scores = self._markov_scores(last_reds)
        scores_red += trans_scores * 3.0

        # === 规律2: 周期性（强信号）===
        periodic_scores_red, periodic_scores_blue = self._periodic_scores()
        scores_red += periodic_scores_red * 2.5
        scores_blue += periodic_scores_blue * 2.5

        # === 规律3: 遗漏回补 ===
        omit_scores_red, omit_scores_blue = self._omission_scores()
        scores_red += omit_scores_red * 1.5
        scores_blue += omit_scores_blue * 2.0  # 蓝球遗漏信号更强

        # === 规律4: 号码对共现 ===
        pair_scores = self._pair_scores(last_reds)
        scores_red += pair_scores * 1.5

        # === 规律5: 热冷交替 ===
        hotcold_scores = self._hot_cold_scores()
        scores_red += hotcold_scores * 0.8

        # === 规律6: 蓝球偏移规律（蓝球差值+1概率最高）===
        blue_drift_scores = self._blue_drift_scores()
        scores_blue += blue_drift_scores * 2.0

        # 归一化
        scores_red = scores_red / (scores_red.sum() + 1e-10)
        scores_blue = scores_blue / (scores_blue.sum() + 1e-10)

        # 生成推荐
        recommendations = []
        sorted_red = np.argsort(scores_red)[::-1]
        sorted_blue = np.argsort(scores_blue)[::-1]

        # 方案A: 综合评分最高
        red_a = sorted(int(r + 1) for r in sorted_red[:RED_COUNT])
        blue_a = int(sorted_blue[0] + 1)
        recommendations.append({"red": red_a, "blue": blue_a, "method": "综合评分"})

        # 方案B: 周期+转移优先
        red_b = sorted(int(r + 1) for r in sorted_red[:4]) + \
                sorted(int(r + 1) for r in np.argsort(periodic_scores_red)[::-1][:3])
        red_b = sorted(list(set(red_b)))[:6]
        blue_b = int(np.argsort(periodic_scores_blue)[::-1][0] + 1)
        recommendations.append({"red": red_b, "blue": blue_b, "method": "周期+转移"})

        # 方案C: 遗漏回补
        red_c = sorted(int(r + 1) for r in np.argsort(omit_scores_red)[::-1][:6])
        blue_c = int(np.argsort(omit_scores_blue)[::-1][0] + 1)
        recommendations.append({"red": red_c, "blue": blue_c, "method": "遗漏回补"})

        return recommendations

    def _markov_scores(self, last_reds):
        """马尔可夫转移得分"""
        scores = np.zeros(RED_MAX)
        recent = self.draws[-300:]
        base_prob = 6 / RED_MAX

        for prev_num in last_reds:
            # 统计 prev_num 出现后各号码的出现率
            trans_count = np.zeros(RED_MAX)
            total = 0
            for idx in range(len(recent) - 1):
                if prev_num in _reds(recent[idx]):
                    total += 1
                    for nr in _reds(recent[idx + 1]):
                        trans_count[nr - 1] += 1
            if total >= 10:
                prob = trans_count / total
                # 得分 = 相对于基准的偏移
                scores += (prob - base_prob)

        return np.clip(scores, 0, None)  # 只保留正向信号

    def _periodic_scores(self):
        """周期性得分"""
        red_scores = np.zeros(RED_MAX)
        blue_scores = np.zeros(BLUE_MAX)
        n = len(self.draws)
        last_idx = n - 1

        for num in range(1, RED_MAX + 1):
            positions = [idx for idx, d in enumerate(self.draws) if num in _reds(d)]
            if len(positions) < 5:
                continue
            intervals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
            if len(intervals) >= 5:
                # 用最近间隔的加权平均作为周期估计
                recent_int = intervals[-20:]
                period = np.mean(recent_int)
                omit = last_idx - positions[-1]
                # 遗漏值接近周期时得分高
                if period > 0:
                    proximity = 1 - abs(omit - period) / period
                    red_scores[num - 1] = max(0, proximity)

        for num in range(1, BLUE_MAX + 1):
            positions = [idx for idx, d in enumerate(self.draws) if d["blue"] == num]
            if len(positions) < 5:
                continue
            intervals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
            if len(intervals) >= 5:
                recent_int = intervals[-20:]
                period = np.mean(recent_int)
                omit = last_idx - positions[-1]
                if period > 0:
                    proximity = 1 - abs(omit - period) / period
                    blue_scores[num - 1] = max(0, proximity)

        return red_scores / (red_scores.sum() + 1e-10), blue_scores / (blue_scores.sum() + 1e-10)

    def _omission_scores(self):
        """遗漏回补得分"""
        n = len(self.draws)
        last_idx = n - 1
        red_omits = np.full(RED_MAX, n, dtype=float)
        blue_omits = np.full(BLUE_MAX, n, dtype=float)

        for idx in range(n - 1, max(0, n - 200) - 1, -1):
            for r in _reds(self.draws[idx]):
                if red_omits[r - 1] == n:
                    red_omits[r - 1] = last_idx - idx
            b = self.draws[idx]["blue"]
            if blue_omits[b - 1] == n:
                blue_omits[b - 1] = last_idx - idx

        # 遗漏值越大，得分越高（回补理论）
        red_scores = red_omits / (red_omits.sum() + 1e-10)
        blue_scores = blue_omits / (blue_omits.sum() + 1e-10)
        return red_scores, blue_scores

    def _pair_scores(self, last_reds):
        """号码对共现得分"""
        scores = np.zeros(RED_MAX)
        recent = self.draws[-200:]
        pair_count = Counter()
        for d in recent:
            reds = sorted(_reds(d))
            for a, b in combinations(reds, 2):
                pair_count[(a, b)] += 1

        # 上期出现的号码，找高频共现伙伴
        for r in last_reds:
            for (a, b), cnt in pair_count.items():
                if a == r:
                    scores[b - 1] += cnt
                elif b == r:
                    scores[a - 1] += cnt

        return scores / (scores.sum() + 1e-10)

    def _hot_cold_scores(self):
        """热冷交替得分"""
        recent = self.draws[-30:]
        freq = Counter()
        for d in recent:
            for r in _reds(d):
                freq[r] += 1

        # 上期热号→本期倾向冷号，上期冷号→本期倾向热号
        last_reds = set(_reds(self.draws[-1]))
        scores = np.zeros(RED_MAX)

        avg_freq = 6 * 30 / RED_MAX
        for num in range(1, RED_MAX + 1):
            f = freq.get(num, 0)
            if num in last_reds:
                scores[num - 1] = max(0, avg_freq - f) / avg_freq
            else:
                scores[num - 1] = f / avg_freq

        return scores / (scores.sum() + 1e-10)

    def _blue_drift_scores(self):
        """蓝球偏移规律：分析上期蓝球→本期蓝球的差值分布"""
        scores = np.zeros(BLUE_MAX)
        recent = self.draws[-200:]
        last_blue = self.draws[-1]["blue"]

        # 统计 last_blue 出现后，下期蓝球的分布
        blue_after = Counter()
        total = 0
        for idx in range(len(recent) - 1):
            if recent[idx]["blue"] == last_blue:
                total += 1
                blue_after[recent[idx + 1]["blue"]] += 1

        if total >= 3:
            for b, cnt in blue_after.items():
                scores[b - 1] = cnt / total
        else:
            # 样本不足，用全局遗漏
            for b in range(1, BLUE_MAX + 1):
                for idx in range(len(recent) - 1, -1, -1):
                    if recent[idx]["blue"] == b:
                        scores[b - 1] = len(recent) - 1 - idx
                        break

        return scores / (scores.sum() + 1e-10)


# ============================================================
# 滚动回测
# ============================================================
def pattern_backtest(draws, start=100, limit=50):
    """用规律预测器做滚动回测"""
    end = len(draws)
    actual_start = max(start, end - limit)

    print(f"\n{'='*50}")
    print(f"【规律预测器滚动回测】第{actual_start+1}期~第{end}期, 共{end-actual_start}期")
    print(f"{'='*50}")

    red_hit_dist = {i: 0 for i in range(7)}
    blue_hits = 0
    total = end - actual_start

    for i in range(actual_start, end):
        train_draws = draws[:i]
        actual = draws[i]
        actual_reds = set(_reds(actual))

        pred = PatternPredictor(train_draws)
        recs = pred.predict(n_sets=1)
        pred_reds = set(recs[0]["red"])
        pred_blue = recs[0]["blue"]

        rh = len(pred_reds & actual_reds)
        bh = 1 if pred_blue == actual["blue"] else 0

        red_hit_dist[rh] += 1
        blue_hits += bh

        done = i - actual_start + 1
        if done % 10 == 0 or done == total:
            avg_r = sum(k * v for k, v in red_hit_dist.items()) / done
            avg_b = blue_hits / done * 100
            print(f"  [{done}/{total}] 第{actual['draw_no']}: "
                  f"红{rh}球 蓝{'✓' if bh else '✗'} | 累计红{avg_r:.2f}/6 蓝{avg_b:.1f}%")

    print(f"\n红球命中分布:")
    for hits, count in red_hit_dist.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {hits}球: {count:4d}期 ({pct:5.1f}%) {bar}")

    avg_red = sum(k * v for k, v in red_hit_dist.items()) / total
    print(f"\n红球平均命中: {avg_red:.2f}/6 (随机期望: {6*6/33:.2f})")
    print(f"蓝球命中率: {blue_hits/total*100:.1f}% (随机期望: {100/16:.1f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true", help="运行规律分析")
    parser.add_argument("--backtest", action="store_true", help="规律预测器回测")
    parser.add_argument("--predict", action="store_true", help="用规律预测下一期")
    parser.add_argument("--limit", type=int, default=50, help="回测期数")
    args = parser.parse_args()

    init_db()
    draws = get_all_draws()
    print(f"总数据: {len(draws)} 期")

    if args.analyze or (not args.backtest and not args.predict):
        pair_cooccurrence(draws)
        markov_analysis(draws)
        periodicity_analysis(draws)
        gap_pattern_analysis(draws)
        zone_rotation_analysis(draws)
        blue_pattern_analysis(draws)

    if args.backtest:
        pattern_backtest(draws, limit=args.limit)

    if args.predict:
        pred = PatternPredictor(draws)
        recs = pred.predict(n_sets=3)
        print(f"\n{'='*50}")
        print(f"规律预测器推荐（第{draws[-1]['draw_no']}期之后）")
        print(f"{'='*50}")
        for i, r in enumerate(recs):
            reds = " ".join(f"{n:02d}" for n in sorted(r["red"]))
            print(f"  方案{chr(65+i)} [{r['method']}]: 红球 {reds} 蓝球 {r['blue']:02d}")
