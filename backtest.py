"""滚动回测：从历史数据中逐期预测，对比实际结果"""
import numpy as np
from ssq.db import init_db, get_all_draws
from ssq.config import RED_MAX, BLUE_MAX
from ssq.features import build_feature_matrix, build_next_features
from ssq.models.baseline import BaselineModel
from ssq.models.xgb_model import XGBPredictor
from ssq.models.lstm_model import LSTMPredictor, build_lstm_input
from ssq.models.ensemble import EnsembleModel
from ssq.baseline import random_baseline_backtest, format_random_baseline


def _predict_probs(draws: list[dict], target_idx: int, include_lstm: bool = False,
                   seq_len: int = 20) -> dict:
    """用 draws[:target_idx] 训练，预测第 target_idx 期，返回各模型概率

    对齐契约：特征/标签只用 draws[:target_idx]，目标期开奖不得进入训练输入。
    Returns: {"baseline": (red33, blue16), "xgboost": ..., "lstm": ...}（仅含可用模型）
    """
    train_draws = draws[:target_idx]

    # 基线模型
    baseline = BaselineModel()
    baseline.fit(train_draws)
    red_base, blue_base = baseline.predict_proba()
    predictions = {"baseline": (red_base, blue_base)}

    # 特征矩阵（训练行 = 全部 minus 1，最后一行特征取自倒数第2期、标签为倒数第1期）
    X, y_red, y_blue = build_feature_matrix(train_draws)
    X_next = build_next_features(train_draws)

    if len(X) >= 50:
        xgb = XGBPredictor()
        xgb.fit(X, y_red, y_blue)
        red_xgb, blue_xgb = xgb.predict_proba(X_next)
        predictions["xgboost"] = (red_xgb, blue_xgb)

    if include_lstm:
        lstm = LSTMPredictor(seq_len=seq_len)
        lstm.fit(X, y_red, y_blue)
        if lstm.model is not None and lstm._available:
            red_lstm, blue_lstm = lstm.predict_proba(build_lstm_input(X, train_draws, seq_len))
            predictions["lstm"] = (red_lstm, blue_lstm)

    return predictions


def walk_forward_backtest(draws: list[dict], start: int = 100, end: int = None,
                          include_lstm: bool = True, dump_path: str = None):
    """
    滚动回测：对每期用之前数据训练，预测当期，对比实际

    Args:
        draws: 全部开奖数据
        start: 起始期索引（需要足够历史训练）
        end: 结束期索引（None=到最后）
        include_lstm: 是否在回测中训练/融合 LSTM（生产口径默认含）
        dump_path: 导出各模型逐期概率的 .npz 路径（供离线调融合权重）
    """
    if end is None:
        end = len(draws)

    total = end - start
    print(f"滚动回测: 第{start+1}期 ~ 第{end}期, 共{total}期")
    print(f"每期用前N期数据训练 → 预测第N+1期 → 对比实际 (LSTM={'开' if include_lstm else '关'})")
    print("=" * 60)

    # 统计
    red_hit_dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    blue_hits = 0
    results = []

    model_names = None
    red_probs_all, blue_probs_all = [], []

    for i in range(start, end):
        predictions = _predict_probs(draws, i, include_lstm=include_lstm)

        if model_names is None:
            model_names = list(predictions.keys())

        for name in model_names:
            if name not in predictions:
                predictions[name] = None  # 某期某模型不可用，概率记 None

        red_probs_all.append([None if predictions[n] is None else predictions[n][0] for n in model_names])
        blue_probs_all.append([None if predictions[n] is None else predictions[n][1] for n in model_names])

        # 融合（概率 None 的模型跳过）
        red_probs, blue_probs = EnsembleModel().combine(
            {n: p for n, p in predictions.items() if p is not None})
        actual = draws[i]
        actual_red = {actual["red1"], actual["red2"], actual["red3"],
                      actual["red4"], actual["red5"], actual["red6"]}
        top6_red = set(np.argsort(red_probs)[-6:] + 1)  # 索引转球号（1-based）再求交集
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
            mark = "Y" if blue_hit else "N"
            print(f"  [{done}/{total}] 第{draw_no}期: "
                  f"红{red_hits}球 蓝{mark} | "
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

    # 随机基线对照
    random_results = random_baseline_backtest(draws, start=start, end=end)
    print(format_random_baseline(random_results))

    # 对比结论
    print("\n" + "=" * 55)
    print("【模型 vs 随机基线】")
    print("=" * 55)
    print(f"  红球平均命中: 模型 {avg_red:.2f} / 随机 {random_results['avg_red']:.2f} / 理论 {6*6/33:.2f}")
    print(f"  蓝球命中率:   模型 {blue_hits/total*100:.1f}% / 随机 {random_results['avg_blue_pct']:.1f}% / 理论 {100/16:.1f}%")
    red_diff = avg_red - random_results["avg_red"]
    blue_diff = blue_hits/total*100 - random_results["avg_blue_pct"]
    print(f"  模型相对随机: 红球 {red_diff:+.2f}, 蓝球 {blue_diff:+.1f}%")

    # 导出各模型逐期概率（供离线调融合权重）
    if dump_path:
        # 概率 None 的模型填 0，另存 mask
        def _stack(per_model):
            arr = np.zeros((len(per_model), len(model_names), per_model[0][0].shape[0]
                            if per_model[0][0] is not None else 1))
            mask = np.ones(arr.shape[:2], dtype=bool)
            for t, row in enumerate(per_model):
                for m, p in enumerate(row):
                    if p is None:
                        mask[t, m] = False
                    else:
                        arr[t, m] = p
            return arr, mask

        red_arr, red_mask = _stack(red_probs_all)
        blue_arr, blue_mask = _stack(blue_probs_all)
        np.savez(dump_path,
                 model_names=np.array(model_names),
                 red_probs=red_arr, red_mask=red_mask,
                 blue_probs=blue_arr, blue_mask=blue_mask,
                 actual_red=np.array([[1.0 if (j + 1) in set(_actual_reds_of(r)) else 0.0
                                       for j in range(RED_MAX)] for r in results]),
                 actual_blue=np.array([float(r["actual_blue"]) for r in results]))
        print(f"已导出逐期概率: {dump_path}")

    return results


def _actual_reds_of(r):
    return set(r["actual_reds"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="滚动回测")
    parser.add_argument("--start", type=int, default=100, help="起始期索引（默认100）")
    parser.add_argument("--end", type=int, default=None, help="结束期索引（默认到最后）")
    parser.add_argument("--limit", type=int, default=None, help="只回测最近N期")
    parser.add_argument("--no-lstm", action="store_true", help="回测不融合LSTM")
    parser.add_argument("--dump", type=str, default=None, help="导出逐期概率的.npz路径")
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

    walk_forward_backtest(draws, start=start, end=end,
                          include_lstm=not args.no_lstm, dump_path=args.dump)
