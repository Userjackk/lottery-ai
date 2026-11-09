"""离线调融合权重/规则：在滚动回测导出的逐期概率上搜索/验证

方法：前半期搜（红球命中/蓝球命中率），后半期验证。只在验证集一致改善才落地。
融合语义与生产 EnsembleModel.combine 一致：先逐模型归一化，再加权求和。
用法：python tune_weights.py --dump /path/to/probs.npz
"""
import argparse
import itertools
import numpy as np


def load_probs(path):
    z = np.load(path, allow_pickle=False)
    names = [str(n) for n in z["model_names"]]
    red, red_mask = z["red_probs"], z["red_mask"]
    blue, blue_mask = z["blue_probs"], z["blue_mask"]
    actual_red, actual_blue = z["actual_red"], z["actual_blue"]
    return names, red, red_mask, blue, blue_mask, actual_red, actual_blue


def _normalize(probs, mask):
    """逐模型归一化（不可用模型置0），与生产 combine 语义一致"""
    p = np.where(mask[:, :, None], probs, 0.0)
    s = p.sum(axis=1, keepdims=True)
    return p / np.clip(s, 1e-10, None)


def _to_ranks(probs, mask):
    """逐模型把概率转成排名分 (33-rank)/(33-1)，排名越高分越高"""
    n_models, _, k = probs.shape
    ranks = np.zeros_like(probs)
    for m in range(n_models):
        if not mask[m]:
            continue
        order = np.argsort(np.argsort(probs[m]))
        ranks[m] = order / (k - 1)
    return ranks


def fuse(red, red_mask, w, rule, target_idx, blue=None, blue_mask=None):
    """按规则融合红球概率，返回 (top6命中数 或 概率向量)"""
    if rule == "avg":
        p = (w[:, None] * red[target_idx]).sum(axis=0)
        return p / p.sum()
    if rule == "rank":
        r = _to_ranks(red, red_mask[target_idx] > 0)
        p = (w[:, None] * r).sum(axis=0)
        return p / p.sum()
    raise ValueError(f"未知规则: {rule}")


def eval_rule(names, red, red_mask, blue, blue_mask, actual_red, actual_blue,
              w_map, rule, lo=0, hi=None):
    """按权重+规则融合并评估红球命中/蓝球命中率"""
    hi = hi or len(actual_red)
    w = np.array([w_map.get(n, 0.0) for n in names])

    red_n = _normalize(red, red_mask)
    blue_n = _normalize(blue, blue_mask)

    scores, blues = [], []
    for t in range(lo, hi):
        m = red_mask[t] * (w > 0)
        if not m.any():
            continue
        p = (w[:, None] * red_n[t] * m[:, None]).sum(axis=0)
        p = p / p.sum()
        top6 = np.argsort(p)[-6:]
        scores.append(int(actual_red[t][top6].astype(bool).sum()))

        mb = blue_mask[t] * (w > 0)
        if not mb.any():
            continue
        pb = (w[:, None] * blue_n[t] * mb[:, None]).sum(axis=0)
        pred = int(np.argmax(pb))
        blues.append(1 if pred + 1 == actual_blue[t] else 0)

    return (np.mean(scores) if scores else 0.0,
            np.mean(blues) * 100 if blues else 0.0, len(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--grid-step", type=float, default=0.05)
    args = ap.parse_args()

    names, red, red_mask, blue, blue_mask, actual_red, actual_blue = load_probs(args.dump)
    n = len(actual_red)
    half = n // 2
    k = len(names)
    print(f"模型: {names}  总期数: {n}（前{half}搜 / 后{n-half}验）")

    # 各模型单独表现（排名与归一化无关，直接 top6）
    print("\n各模型单独表现（全部期数）:")
    for m_idx, m in enumerate(names):
        hits, blues = [], []
        for t in range(n):
            if not red_mask[t, m_idx]:
                continue
            top6 = np.argsort(red[t, m_idx])[-6:]
            hits.append(int(actual_red[t][top6].astype(bool).sum()))
            pred_b = int(np.argmax(blue[t, m_idx])) + 1
            blues.append(1 if pred_b == actual_blue[t] else 0)
        print(f"  {m:10s} 红 {np.mean(hits):.3f}/6 ({len(hits)}期)  蓝 {np.mean(blues)*100:.1f}%")

    # 网格搜索（权重和为 1）
    step = args.grid_step
    n_steps = int(round(1.0 / step))
    combos = []
    for parts in itertools.product(range(n_steps + 1), repeat=k):
        if sum(parts) != n_steps:
            continue
        w_map = {m: p * step for m, p in zip(names, parts)}
        avg_red, avg_blue, _ = eval_rule(names, red, red_mask, blue, blue_mask,
                                         actual_red, actual_blue, w_map, "avg", 0, half)
        combos.append((avg_red, avg_blue, w_map))

    combos.sort(key=lambda c: c[0], reverse=True)
    print(f"\n搜索集 TOP5（加权平均, 按红球平均命中）:")
    for avg_red, avg_blue, w_map in combos[:5]:
        w_str = " ".join(f"{m}={w_map[m]:.2f}" for m in names)
        print(f"  红{avg_red:.3f} 蓝{avg_blue:.1f}%  [{w_str}]")

    # 验证集评估：搜索 top5 + 当前默认 + 等权
    print("\n验证集对照（加权平均）:")
    candidates = [("默认", {m: w for m, w in zip(names, [0.25, 0.45, 0.3][:k])}),
                  ("等权", {m: 1.0 / k for m in names})]
    for i, (_, _, w_map) in enumerate(combos[:5]):
        candidates.append((f"搜索TOP{i+1}", w_map))

    for label, w_map in candidates:
        avg_red, avg_blue, _ = eval_rule(names, red, red_mask, blue, blue_mask,
                                         actual_red, actual_blue, w_map, "avg", half, n)
        w_str = " ".join(f"{m}={w_map.get(m, 0):.2f}" for m in names)
        print(f"  {label:10s} 红{avg_red:.3f} 蓝{avg_blue:.1f}%  [{w_str}]")

    # 最优权重下：排名融合 vs 加权平均（验证集）
    best_map = combos[0][2]
    print("\n融合规则对比（验证集, 搜索TOP1权重）:")
    for rule in ["avg", "rank"]:
        avg_red, avg_blue, _ = eval_rule(names, red, red_mask, blue, blue_mask,
                                         actual_red, actual_blue, best_map, rule, half, n)
        print(f"  {rule:5s} 红{avg_red:.3f} 蓝{avg_blue:.1f}%")

    # 约束过滤采样 vs 纯 top6（验证集, 默认权重, 加权平均）
    from ssq.constraints import ConstraintFilter
    cf = ConstraintFilter([])
    default_w = {m: w for m, w in zip(names, [0.25, 0.45, 0.3][:k])}
    red_n = _normalize(red, red_mask)
    raw_hits, cons_hits = [], []
    rng = np.random.default_rng(0)
    for t in range(half, n):
        w = np.array([default_w.get(m, 0.0) for m in names])
        p = (w[:, None] * red_n[t] * red_mask[t][:, None]).sum(axis=0)
        if p.sum() <= 0:
            continue
        p = p / p.sum()
        top6 = np.argsort(p)[-6:]
        raw_hits.append(int(actual_red[t][top6].astype(bool).sum()))
        combo = None
        for _ in range(2000):
            cand = sorted((rng.choice(33, size=6, replace=False, p=p) + 1).tolist())
            if cf.check(cand):
                combo = cand
                break
        if combo is None:
            combo = sorted((top6 + 1).tolist())
        cons_hits.append(len(set(combo) & {j + 1 for j in np.where(actual_red[t] == 1)[0]}))
    if raw_hits:
        print(f"\n约束过滤影响（验证集, 默认权重, 加权平均）:")
        print(f"  纯 top6:      红 {np.mean(raw_hits):.3f}/6")
        print(f"  约束采样 top6: 红 {np.mean(cons_hits):.3f}/6")


if __name__ == "__main__":
    main()
