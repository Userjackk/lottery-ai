"""策略选号：基于统计约束生成"合理"组合，不做概率预测"""
import numpy as np
from collections import Counter
from itertools import combinations
from ssq.db import init_db, get_all_draws
from ssq.config import RED_MAX, BLUE_MAX, RED_COUNT


def _reds(d):
    return [d["red1"], d["red2"], d["red3"], d["red4"], d["red5"], d["red6"]]


class StrategyPicker:
    """策略选号器：基于历史统计约束生成号码组合"""

    def __init__(self, draws: list[dict]):
        self.draws = draws
        self.n = len(draws)

    # ============================================================
    # 统计约束计算
    # ============================================================

    def _zone_dist(self, window=200):
        """三区分布统计：返回最常见分布"""
        recent = self.draws[-window:]
        dists = Counter()
        for d in recent:
            reds = _reds(d)
            z1 = sum(1 for r in reds if r <= 11)
            z2 = sum(1 for r in reds if 12 <= r <= 22)
            z3 = sum(1 for r in reds if r >= 23)
            dists[(z1, z2, z3)] += 1
        return dists

    def _odd_even_dist(self, window=200):
        """奇偶比统计"""
        recent = self.draws[-window:]
        dists = Counter()
        for d in recent:
            reds = _reds(d)
            odd = sum(1 for r in reds if r % 2 == 1)
            dists[f"{odd}:{6-odd}"] += 1
        return dists

    def _sum_range(self, window=200):
        """红球和值范围"""
        recent = self.draws[-window:]
        sums = [sum(_reds(d)) for d in recent]
        return {
            "min": min(sums),
            "max": max(sums),
            "mean": np.mean(sums),
            "std": np.std(sums),
            "p10": np.percentile(sums, 10),
            "p90": np.percentile(sums, 90),
        }

    def _gap_stats(self, window=200):
        """间距统计"""
        recent = self.draws[-window:]
        all_gaps = []
        for d in recent:
            reds = sorted(_reds(d))
            gaps = [reds[i+1] - reds[i] for i in range(5)]
            all_gaps.append(gaps)
        return {
            "mean": np.mean(all_gaps, axis=0),
            "std": np.std(all_gaps, axis=0),
        }

    def _hot_numbers(self, window=30, threshold=1.3):
        """热号：近期高频出现"""
        recent = self.draws[-window:]
        freq = Counter()
        for d in recent:
            for r in _reds(d):
                freq[r] += 1
        avg = 6 * window / RED_MAX
        return [n for n, f in freq.items() if f >= avg * threshold]

    def _cold_numbers(self, window=100, threshold=2.0):
        """冷号：长期遗漏"""
        recent = self.draws[-window:]
        last_seen = {}
        for idx, d in enumerate(recent):
            for r in _reds(d):
                last_seen[r] = idx
        cold = []
        for n in range(1, RED_MAX + 1):
            omit = (len(recent) - 1) - last_seen.get(n, -1)
            if omit >= len(recent) * threshold / RED_MAX:
                cold.append(n)
        return cold

    def _consecutive_tendency(self, window=200):
        """连号倾向：历史有连号的概率"""
        recent = self.draws[-window:]
        has_consec = 0
        for d in recent:
            reds = sorted(_reds(d))
            if any(reds[i+1] - reds[i] == 1 for i in range(5)):
                has_consec += 1
        return has_consec / len(recent)

    def _blue_stats(self, window=200):
        """蓝球统计"""
        recent = self.draws[-window:]
        # 遗漏值
        blue_omit = {}
        for b in range(1, BLUE_MAX + 1):
            for idx in range(len(recent) - 1, -1, -1):
                if recent[idx]["blue"] == b:
                    blue_omit[b] = len(recent) - 1 - idx
                    break
            else:
                blue_omit[b] = len(recent)

        # 上期蓝球偏移
        last_blue = recent[-1]["blue"]
        blue_after = Counter()
        total = 0
        for idx in range(len(recent) - 1):
            if recent[idx]["blue"] == last_blue:
                total += 1
                blue_after[recent[idx + 1]["blue"]] += 1

        return {
            "omit": blue_omit,
            "last": last_blue,
            "after": blue_after,
            "after_total": total,
        }

    # ============================================================
    # 约束过滤器
    # ============================================================

    def _check_zone(self, reds):
        """检查区间分布是否合理"""
        z1 = sum(1 for r in reds if r <= 11)
        z2 = sum(1 for r in reds if 12 <= r <= 22)
        z3 = sum(1 for r in reds if r >= 23)
        dist = (z1, z2, z3)
        # 最常见分布：[2,2,2], [3,1,2], [1,3,2], [3,2,1], [1,2,3]
        valid = {(2,2,2), (3,1,2), (1,3,2), (3,2,1), (1,2,3),
                 (2,3,1), (2,1,3), (2,2,2), (1,4,1), (4,1,1)}
        return dist in valid

    def _check_odd_even(self, reds):
        """检查奇偶比是否合理"""
        odd = sum(1 for r in reds if r % 2 == 1)
        ratio = f"{odd}:{6-odd}"
        # 最常见：3:3, 4:2, 2:4
        return ratio in {"3:3", "4:2", "2:4"}

    def _check_sum(self, reds):
        """检查和值是否在合理范围"""
        sr = self._sum_range()
        s = sum(reds)
        return sr["p10"] <= s <= sr["p90"]

    def _check_consecutive(self, reds, max_pairs=2):
        """检查连号数量是否合理"""
        reds = sorted(reds)
        pairs = sum(1 for i in range(5) if reds[i+1] - reds[i] == 1)
        return pairs <= max_pairs

    def _check_gap(self, reds):
        """检查间距是否合理"""
        reds = sorted(reds)
        gaps = [reds[i+1] - reds[i] for i in range(5)]
        # 全部间距为1（全连号）或太大都不合理
        if all(g == 1 for g in gaps):
            return False
        if any(g > 15 for g in gaps):
            return False
        return True

    def _passes_all(self, reds):
        """通过所有约束"""
        return (self._check_zone(reds) and
                self._check_odd_even(reds) and
                self._check_sum(reds) and
                self._check_consecutive(reds) and
                self._check_gap(reds))

    # ============================================================
    # 蓝球策略
    # ============================================================

    def _pick_blue(self, strategy="omit"):
        """蓝球选号策略"""
        blue_stats = self._blue_stats()

        if strategy == "omit":
            # 高遗漏回补：选遗漏值最大的
            sorted_blue = sorted(blue_stats["omit"].items(), key=lambda x: x[1], reverse=True)
            return sorted_blue[0][0]

        elif strategy == "drift":
            # 偏移策略：基于上期蓝球的转移概率
            after = blue_stats["after"]
            total = blue_stats["after_total"]
            if total >= 3:
                best = max(after.items(), key=lambda x: x[1])
                return best[0]
            # 样本不足，选遗漏最高的
            return self._pick_blue("omit")

        elif strategy == "balanced":
            # 均衡策略：遗漏适中 + 偏移信号
            omit = blue_stats["omit"]
            after = blue_stats["after"]
            total = blue_stats["after_total"]

            scores = {}
            avg_omit = self.n / BLUE_MAX
            for b in range(1, BLUE_MAX + 1):
                # 遗漏得分：遗漏越大越好，但不要太极端
                omit_score = min(omit.get(b, 0) / avg_omit, 3.0)
                # 偏移得分
                drift_score = after.get(b, 0) / max(total, 1) if total >= 3 else 0
                scores[b] = omit_score * 0.6 + drift_score * 0.4

            return max(scores, key=scores.get)

    # ============================================================
    # 主策略：生成推荐
    # ============================================================

    def generate(self, n_sets=3, max_attempts=10000):
        """基于统计约束生成推荐号码"""
        # 统计约束参数
        hot = self._hot_numbers()
        cold = self._cold_numbers()
        consec_tendency = self._consecutive_tendency()
        blue_stats = self._blue_stats()

        # 收集通过约束的组合
        valid_combos = []

        # 生成候选：混合热号、冷号、随机
        candidates = []

        # 策略1：热号为主 + 冷号点缀
        for _ in range(max_attempts // 3):
            n_hot = np.random.choice([3, 4])
            n_cold = np.random.choice([0, 1, 2])
            n_random = RED_COUNT - n_hot - n_cold
            if n_random < 0:
                continue

            picked = list(np.random.choice(hot, size=min(n_hot, len(hot)), replace=False)) if hot else []
            remaining_cold = [c for c in cold if c not in picked]
            if remaining_cold and n_cold > 0:
                picked += list(np.random.choice(remaining_cold, size=min(n_cold, len(remaining_cold)), replace=False))
            remaining = [r for r in range(1, RED_MAX + 1) if r not in picked]
            if len(picked) < RED_COUNT and remaining:
                picked += list(np.random.choice(remaining, size=RED_COUNT - len(picked), replace=False))

            if len(set(picked)) == RED_COUNT:
                candidates.append(sorted(set(picked)))

        # 策略2：区间均衡选号
        for _ in range(max_attempts // 3):
            z1 = sorted(np.random.choice(range(1, 12), size=np.random.choice([2, 3]), replace=False))
            z2 = sorted(np.random.choice(range(12, 23), size=np.random.choice([1, 2, 3]), replace=False))
            z3 = sorted(np.random.choice(range(23, 34), size=np.random.choice([1, 2, 3]), replace=False))
            picked = sorted(list(z1) + list(z2) + list(z3))
            if len(picked) == RED_COUNT:
                candidates.append(picked)

        # 策略3：奇偶均衡选号
        for _ in range(max_attempts // 3):
            odds = list(np.random.choice(range(1, 34, 2), size=3, replace=False))
            evens = list(np.random.choice(range(2, 34, 2), size=3, replace=False))
            picked = sorted(odds + evens)
            if len(picked) == RED_COUNT:
                candidates.append(picked)

        # 约束过滤
        for combo in candidates:
            if self._passes_all(combo):
                valid_combos.append(combo)

        # 去重
        valid_combos = [list(x) for x in set(tuple(x) for x in valid_combos)]

        if not valid_combos:
            # 约束太严，放松后重试
            return self._fallback_generate(n_sets)

        # 按"合理度"排序：热号数量 + 和值接近均值
        sr = self._sum_range()
        target_sum = sr["mean"]

        def score(combo):
            hot_count = sum(1 for r in combo if r in hot)
            cold_count = sum(1 for r in combo if r in cold)
            sum_penalty = abs(sum(combo) - target_sum) / 50
            return hot_count * 2 + cold_count * 1 - sum_penalty

        valid_combos.sort(key=score, reverse=True)

        # 取前 n_sets 个（加一些随机性避免每次都一样）
        results = []
        top_pool = valid_combos[:max(n_sets * 5, 20)]
        indices = np.random.choice(len(top_pool), size=min(n_sets, len(top_pool)), replace=False)
        for idx in indices:
            combo = valid_combos[idx]

            # 蓝球策略
            blue_strategies = ["omit", "drift", "balanced"]
            blue = self._pick_blue(blue_strategies[len(results) % 3])

            results.append({
                "red": combo,
                "blue": blue,
            })

        return results

    def _fallback_generate(self, n_sets):
        """约束放松的备用生成"""
        results = []
        for i in range(n_sets):
            # 基本约束：区间均衡 + 奇偶合理
            for _ in range(1000):
                z1 = sorted(np.random.choice(range(1, 12), size=2, replace=False))
                z2 = sorted(np.random.choice(range(12, 23), size=2, replace=False))
                z3 = sorted(np.random.choice(range(23, 34), size=2, replace=False))
                reds = sorted(list(z1) + list(z2) + list(z3))
                odd = sum(1 for r in reds if r % 2 == 1)
                if 2 <= odd <= 4:
                    blue = self._pick_blue(["omit", "drift", "balanced"][i % 3])
                    results.append({"red": reds, "blue": blue})
                    break
        return results


# ============================================================
# 滚动回测
# ============================================================

def strategy_backtest(draws, start=100, limit=100):
    """策略选号滚动回测"""
    end = len(draws)
    actual_start = max(start, end - limit)

    print(f"\n{'='*55}")
    print(f"【策略选号滚动回测】第{actual_start+1}期~第{end}期, 共{end-actual_start}期")
    print(f"{'='*55}")

    red_hit_dist = {i: 0 for i in range(7)}
    blue_hits = 0
    total = end - actual_start

    for i in range(actual_start, end):
        train_draws = draws[:i]
        actual = draws[i]
        actual_reds = set(_reds(actual))

        picker = StrategyPicker(train_draws)
        recs = picker.generate(n_sets=1)
        pred_reds = set(recs[0]["red"])
        pred_blue = recs[0]["blue"]

        rh = len(pred_reds & actual_reds)
        bh = 1 if pred_blue == actual["blue"] else 0

        red_hit_dist[rh] += 1
        blue_hits += bh

        done = i - actual_start + 1
        if done % 20 == 0 or done == total:
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

    # 统计生成的组合特征
    print(f"\n策略约束验证:")
    picker = StrategyPicker(draws[:actual_start])
    combos = picker.generate(n_sets=100, max_attempts=50000)
    if combos:
        sums = [sum(c["red"]) for c in combos]
        odds = [sum(1 for r in c["red"] if r % 2 == 1) for c in combos]
        print(f"  生成{len(combos)}组, 和值: {np.mean(sums):.0f}±{np.std(sums):.0f} (历史均值{np.mean([sum(_reds(d)) for d in draws[-200:]]):.0f})")
        print(f"  奇数个数: {np.mean(odds):.1f}±{np.std(odds):.1f} (历史最常见3.0)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    init_db()
    draws = get_all_draws()
    print(f"总数据: {len(draws)} 期")

    if args.backtest:
        strategy_backtest(draws, limit=args.limit)

    if args.predict:
        picker = StrategyPicker(draws)
        recs = picker.generate(n_sets=3)
        print(f"\n{'='*50}")
        print(f"策略选号推荐（第{draws[-1]['draw_no']}期之后）")
        print(f"{'='*50}")
        for i, r in enumerate(recs):
            reds = " ".join(f"{n:02d}" for n in r["red"])
            z1 = sum(1 for x in r["red"] if x <= 11)
            z2 = sum(1 for x in r["red"] if 12 <= x <= 22)
            z3 = sum(1 for x in r["red"] if x >= 23)
            odd = sum(1 for x in r["red"] if x % 2 == 1)
            print(f"  方案{chr(65+i)}: 红球 {reds} 蓝球 {r['blue']:02d}")
            print(f"    区间[{z1},{z2},{z3}] 奇偶{odd}:{6-odd} 和值{sum(r['red'])}")
