"""滚动回测：从历史数据中逐期预测，对比实际结果"""
import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
from ssq.db import init_db, get_all_draws
from ssq.config import RED_MAX, BLUE_MAX
from ssq.features import build_feature_matrix
from ssq.models.baseline import BaselineModel
from ssq.models.xgb_model import XGBPredictor
from ssq.models.ensemble import EnsembleModel


def walk_forward_backtest(draws: list[dict], start: int = 100, end: int = None):
    """
    滚动回测：对每期用之前数据训练，预测当期，对比实际

    Args:
        draws: 全部开奖数据
        start: 起始期索引（需要足够历史训练）
        end: 结束期索引（None=到最后）
    """
    if end is None:
        end = len(draws)

    total = end - start
    print(f"滚动回测: 第{start+1}期 ~ 第{end}期, 共{total}期")
    print(f"每期用前N期数据训练 → 预测第N+1期 → 对比实际")
    print("=" * 60)

    # 统计
    red_hit_dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    blue_hits = 0
    results = []

    for i in range(start, end):
        # 用前 i 期训练，预测第 i 期
        train_draws = draws[:i]
        actual = draws[i]

        # 构建特征
        X, y_red, y_blue = build_feature_matrix(train_draws)
        X_next = X[-1:]  # 用最后一行作为"下一期"特征（实际是第i期的特征）

        # 基线模型
        baseline = BaselineModel()
        baseline.fit(train_draws)
        red_base, blue_base = baseline.predict_proba()

        # XGBoost
        xgb = XGBPredictor()
        X_train = X[:-1]
        y_red_train = y_red[:-1]
        y_blue_train = y_blue[:-1]
        if len(X_train) < 50:
            # 数据太少，只用基线
            red_probs = red_base
            blue_probs = blue_base
        else:
            xgb.fit(X_train, y_red_train, y_blue_train)
            red_xgb, blue_xgb = xgb.predict_proba(X_next)

            # 融合
            ens = EnsembleModel()
            predictions = {
                "baseline": (red_base, blue_base),
                "xgboost": (red_xgb, blue_xgb),
            }
            red_probs, blue_probs = ens.combine(predictions)

        # 取 top6 红球
        top6_red = set(np.argsort(red_probs)[-6:])
        actual_red = {actual["red1"], actual["red2"], actual["red3"],
                      actual["red4"], actual["red5"], actual["red6"]}
        red_hits = len(top6_red & actual_red)

        # 蓝球
        pred_blue = int(np.argmax(blue_probs)) + 1
        blue_hit = 1 if pred_blue == actual["blue"] else 0

        red_hit_dist[red_hits] += 1
        blue_hits += blue_hit

        # 记录
        draw_no = actual["draw_no"]
        pred_reds = sorted(int(r + 1) for r in top6_red)
        actual_reds_sorted = sorted(actual_red)
        results.append({
            "draw_no": draw_no,
            "pred_reds": pred_reds,
            "actual_reds": actual_reds_sorted,
            "red_hits": red_hits,
            "pred_blue": pred_blue,
            "actual_blue": actual["blue"],
            "blue_hit": blue_hit,
        })

        # 进度
        done = i - start + 1
        if done % 10 == 0 or done == total:
            avg_red = sum(r["red_hits"] for r in results) / len(results)
            avg_blue = sum(r["blue_hit"] for r in results) / len(results) * 100
            print(f"  [{done}/{total}] 第{draw_no}期: "
                  f"红{red_hits}球 蓝{'✓' if blue_hit else '✗'} | "
                  f"累计红{avg_red:.2f}/6 蓝{avg_blue:.1f}%")

    # 汇总
    print("\n" + "=" * 60)
    print("回测汇总")
    print("=" * 60)
    print(f"总期数: {total}")
    print(f"\n红球命中分布:")
    for hits, count in red_hit_dist.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {hits}球: {count:4d}期 ({pct:5.1f}%) {bar}")

    avg_red = sum(r["red_hits"] for r in results) / total
    print(f"\n红球平均命中: {avg_red:.2f}/6 (理论期望: {6*6/33:.2f})")
    print(f"蓝球命中率: {blue_hits/total*100:.1f}% (理论期望: {100/16:.1f}%)")
    print(f"蓝球命中: {blue_hits}/{total}")

    # 最佳/最差
    best = max(results, key=lambda r: r["red_hits"])
    worst = min(results, key=lambda r: r["red_hits"])
    print(f"\n最佳: 第{best['draw_no']}期 红{best['red_hits']}球 "
          f"预测{best['pred_reds']} 实际{best['actual_reds']}")
    print(f"最差: 第{worst['draw_no']}期 红{worst['red_hits']}球 "
          f"预测{worst['pred_reds']} 实际{worst['actual_reds']}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="滚动回测")
    parser.add_argument("--start", type=int, default=100, help="起始期索引（默认100）")
    parser.add_argument("--end", type=int, default=None, help="结束期索引（默认到最后）")
    parser.add_argument("--limit", type=int, default=None, help="只回测最近N期")
    args = parser.parse_args()

    init_db()
    draws = get_all_draws()
    print(f"总数据: {len(draws)} 期\n")

    end = args.end
    if args.limit and not end:
        end = len(draws)
    if args.limit:
        start = max(args.start, end - args.limit)
    else:
        start = args.start

    walk_forward_backtest(draws, start=start, end=end)
