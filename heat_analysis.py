"""号码热度分析：哪些组合结构特征与彩民购买集中度相关

只做测量，不改选号逻辑。运行方式：
    python -u heat_analysis.py --refresh
"""
import argparse
import sys
from datetime import date

import numpy as np

from ssq import db
from ssq.heat import (
    FEATURE_KEYS, NEGATIVE_CONTROL, bh_correct, build_dataset,
    check_heat_uniformity, ols_hc1, quantile_groups,
)
from ssq.prize_data import (
    fetch_with_prizes, get_all_prize_stats, init_prize_table,
    insert_prize_stats,
)

DEFAULT_OUT = "docs/heat-analysis-2026-11-26.md"


def _year_of(draw_no) -> float:
    try:
        return float(str(draw_no)[:4])
    except ValueError:
        return 0.0


def _weekday_of(date_str) -> float:
    try:
        return float(date.fromisoformat(str(date_str)[:10]).weekday())
    except ValueError:
        return 0.0


def _bootstrap_diff_ci(a: np.ndarray, b: np.ndarray,
                       n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """两组均值差的 bootstrap 置信区间"""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = (rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean())
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(lo), float(hi)


_NONE = "—"


def _feature_stats(rows: list[dict], feature_keys: list[str],
                   n_groups: int = 3) -> list[dict]:
    """每个特征的分档统计 + 多变量 β/p(BH) + 单变量 β/p(BH)"""
    usable = [r for r in rows
              if not np.isnan(float(r.get("h2", np.nan)))
              and (r.get("sales") or 0) > 0]
    if not usable:
        return []

    y = np.log(np.array([float(r["h2"]) for r in usable]))
    cols = [np.log(np.array([float(r["sales"]) for r in usable]))]
    for k in feature_keys:
        cols.append(np.array([float(r.get(k, 0.0)) for r in usable]))
    cols.append(np.array([_year_of(r.get("draw_no", "")) for r in usable]))
    cols.append(np.array([_weekday_of(r.get("date", "")) for r in usable]))

    fit = ols_hc1(np.column_stack(cols), y)
    # X 列 = [log_sales, 各特征..., year, weekday]；beta 额外多一个截距
    # 故第 i 个特征对应 beta/p 的下标 2 + i
    p_adj = bh_correct([fit["p"][2 + i] for i in range(len(feature_keys))])

    log_sales, year_col, wd_col = cols[0], cols[-2], cols[-1]
    uni_beta, uni_p = [], []
    for k in feature_keys:
        f = np.array([float(r.get(k, 0.0)) for r in usable])
        if np.std(f) == 0:
            uni_beta.append(np.nan)
            uni_p.append(np.nan)
            continue
        fit1 = ols_hc1(np.column_stack([log_sales, f, year_col, wd_col]), y)
        uni_beta.append(float(fit1["beta"][2]))
        uni_p.append(float(fit1["p"][2]))
    uni_adj = bh_correct(uni_p)

    stats = []
    for i, k in enumerate(feature_keys):
        beta = float(fit["beta"][2 + i])
        pv = float(p_adj[i])
        vals = np.array([float(r.get(k, 0.0)) for r in usable])
        entry = {
            "key": k, "beta": beta, "p_adj": pv,
            "uni_beta": uni_beta[i], "uni_p": float(uni_adj[i]),
            "agree": True, "means": [_NONE] * n_groups,
            "diff": np.nan, "ci": (np.nan, np.nan), "judge": "无方差",
        }

        if np.std(vals) == 0:
            stats.append(entry)
            continue

        g = quantile_groups(vals, n_groups)
        for gi in range(n_groups):
            sub = y[g == gi]
            if len(sub):
                entry["means"][gi] = f"{sub.mean():+.3f}"

        lo_v, hi_v = y[g == 0], y[g == n_groups - 1]
        # 极度偏斜的特征会出现空分档（如 95% 取 0），此时不报档间差
        if len(lo_v) >= 2 and len(hi_v) >= 2:
            entry["diff"] = float(hi_v.mean() - lo_v.mean())
            entry["ci"] = _bootstrap_diff_ci(hi_v, lo_v)

        if np.isnan(pv):
            entry["judge"] = "无方差"
        elif pv < 0.05 and beta > 0:
            entry["judge"] = "显著(正)"
        elif pv < 0.05 and beta < 0:
            entry["judge"] = "显著(负)"
        else:
            entry["judge"] = "不显著"

        if not np.isnan(uni_beta[i]) and uni_beta[i] * beta < 0:
            entry["agree"] = False
        stats.append(entry)
    return stats


def format_feature_table(rows: list[dict], feature_keys: list[str],
                         n_groups: int = 3) -> str:
    """特征三档 log(H2) 均值 + CI + 多元回归 β/p（BH 校正）+ 单/多变量交叉核对"""
    stats = _feature_stats(rows, feature_keys, n_groups)
    if not stats:
        return "（无可用于分析的行）"

    lines = [
        "| 特征 | 低档 logH2 | 中档 logH2 | 高档 logH2 | 高档-低档 | 95% CI | β | p(BH) | 判定 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in stats:
        diff_s = f"{s['diff']:+.3f}" if not np.isnan(s["diff"]) else _NONE
        ci_s = (f"[{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]"
                if not np.isnan(s["diff"]) else _NONE)
        pv_s = f"{s['p_adj']:.4f}" if not np.isnan(s["p_adj"]) else _NONE
        lines.append(
            f"| {s['key']} | {s['means'][0]} | {s['means'][1]} | {s['means'][2]} | "
            f"{diff_s} | {ci_s} | {s['beta']:+.4f} | {pv_s} | {s['judge']} |"
        )

    lines.append("")
    lines.append("### 交叉核对（单变量 vs 多变量，控制变量相同）")
    lines.append("")
    lines.append("| 特征 | 单变量 β | 单变量 p(BH) | 多变量 β | 一致性 |")
    lines.append("|---|---|---|---|---|")
    for s in stats:
        if np.isnan(s["uni_beta"]):
            lines.append(f"| {s['key']} | {_NONE} | {_NONE} | {s['beta']:+.4f} | 无方差 |")
            continue
        uni_p_s = f"{s['uni_p']:.4g}" if not np.isnan(s["uni_p"]) else _NONE
        agree = "相反(疑共线)" if not s["agree"] else "一致"
        lines.append(
            f"| {s['key']} | {s['uni_beta']:+.4f} | {uni_p_s} | "
            f"{s['beta']:+.4f} | {agree} |"
        )
    return "\n".join(lines)


def format_conclusion(stats: list[dict], uniformity: dict) -> str:
    """结论：有证据的规则、证据不足的规则、阴性对照、期望奖金换算"""
    lines = []

    solid, weak = [], []
    for s in stats:
        if s["judge"] not in ("显著(正)", "显著(负)"):
            continue
        if not s["agree"]:
            weak.append((s, "单/多变量方向相反，疑共线，不可用"))
        elif np.isnan(s["uni_p"]) or s["uni_p"] >= 0.05:
            weak.append((s, "单变量不显著，证据弱"))
        elif np.isnan(s["diff"]):
            weak.append((s, "分档退化，无法给出档间差"))
        elif s["ci"][0] <= 0 <= s["ci"][1]:
            weak.append((s, "分组 95% CI 跨零，证据弱"))
        else:
            solid.append(s)

    lines.append("### 有证据的规则（单/多变量方向一致、单变量亦显著、分组 CI 不含 0）")
    lines.append("")
    if not solid:
        lines.append("无。没有任何特征达到可用证据强度，不应据此改动避号规则。")
    else:
        lines.append("| 特征 | 高档/低档热度 | 处理建议 |")
        lines.append("|---|---|---|")
        for s in solid:
            mult = float(np.exp(s["diff"]))
            advice = ("高档更热，应避开高档" if s["diff"] > 0 else "低档更热，应避开低档")
            lines.append(f"| {s['key']} | {mult:.2f}x | {advice} |")

    lines.append("")
    lines.append("### 证据不足，不应用于选号")
    lines.append("")
    if not weak:
        lines.append("无。")
    else:
        for s, why in weak:
            lines.append(f"- {s['key']}：{why}")

    lines.append("")
    lines.append("### 阴性对照")
    lines.append("")
    ctrl = next((s for s in stats if s["key"] == NEGATIVE_CONTROL), None)
    if ctrl is None:
        lines.append("未包含阴性对照特征。")
    elif ctrl["judge"] == "不显著":
        lines.append(
            f"{NEGATIVE_CONTROL}（由开奖期号数字和推出的奇偶标志，与号码结构天然独立）"
            "不显著 -> 口径无系统偏差。"
        )
    else:
        lines.append(
            f"{NEGATIVE_CONTROL} 被判显著 -> 口径可能有系统偏差，结论需谨慎。"
        )

    lines.append("")
    lines.append("### 换算到期望奖金")
    lines.append("")
    if solid:
        best = max(solid, key=lambda s: abs(s["diff"]))
        mult = float(np.exp(abs(best["diff"])))
        lines.append(
            f"单条规则里最强的是 {best['key']}：高低档之间热度约差 {mult:.2f} 倍。"
            "选热度低的一档，中头奖/二等奖时同分奖金的人相应更少。"
        )
    lines.append(
        "注意：只有一等奖、二等奖浮动（按注数分摊）；三至六等奖固定，与选号无关。"
        "因此这条只在少数井喷期有明显金额差别，不改变中奖概率。"
    )
    if not uniformity.get("ok", True):
        lines.append("")
        lines.append("均匀性自检未通过，以上结论仅供参考。")
    return "\n".join(lines)


def format_report(rows: list[dict], uniformity: dict, feature_table: str,
                  top_n: int = 20, conclusion: str = None) -> str:
    lines = ["# 双色球号码热度实证分析", ""]

    lines.append("## 均匀性自检")
    lines.append(f"- 有效样本：{uniformity['n']} 期")
    lines.append(f"- H2 均值：{uniformity['mean']:.3f}（应为 1）")
    verdict = "通过" if uniformity["ok"] else "未通过 —— 口径可能有误，下面的结论仅供参考"
    lines.append(f"- 判定：{verdict}")
    pairs = [(r["h1"], r["h2"]) for r in rows
             if r.get("h1") is not None and r.get("h2") is not None
             and not np.isnan(r["h1"]) and not np.isnan(r["h2"])]
    if len(pairs) >= 3:
        arr = np.array(pairs)
        if np.std(arr[:, 0]) > 0 and np.std(arr[:, 1]) > 0:
            r_val = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
            lines.append(f"- H1 与 H2 相关：r={r_val:.2f}（应显著为正，两者测同一现象）")
    lines.append("")

    lines.append("## 特征检验")
    lines.append("")
    lines.append(feature_table)
    lines.append("")

    lines.append(f"## 最热 {top_n} 期（按 H2）")
    lines.append("")
    lines.append("| 期号 | H2 | H1 | 红球 | 蓝球 | 一等奖注数 | 二等奖注数 |")
    lines.append("|---|---|---|---|---|---|---|")
    top = sorted(rows, key=lambda r: -(r.get("h2") if r.get("h2") is not None else -1))[:top_n]
    for r in top:
        reds = " ".join(f"{x:02d}" for x in (r.get("reds") or [])) or _NONE
        blue = f"{r['blue']:02d}" if r.get("blue") is not None else _NONE
        lines.append(
            f"| {r['draw_no']} | {r['h2']:.2f} | {r['h1']:.2f} | {reds} | {blue} | "
            f"{r['n1']} | {r['n2']} |"
        )
    lines.append("")

    lines.append("## 结论")
    lines.append("")
    if conclusion is None:
        lines.append("见上表；方向以 β 符号为准。")
    else:
        lines.append(conclusion)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="双色球号码热度实证分析")
    parser.add_argument("--refresh", action="store_true", help="重新抓取奖级数据")
    parser.add_argument("--out", default=DEFAULT_OUT, help="报告输出路径")
    parser.add_argument("--no-write", action="store_true", help="只打印不写文件")
    args = parser.parse_args()

    db.init_db()
    init_prize_table()

    if args.refresh or len(get_all_prize_stats()) < 100:
        print("正在抓取奖级数据（销售额 / 各等奖注数）...")
        fetched = fetch_with_prizes()
        print(f"  抓取 {len(fetched)} 期，写入 {insert_prize_stats(fetched)} 期")

    draws = db.get_all_draws()
    prize = get_all_prize_stats()
    rows = build_dataset(prize, draws)

    denom = min(len(draws), len(prize)) or 1
    print(f"开奖 {len(draws)} 期 / 奖级 {len(prize)} 期 / 匹配 {len(rows)} 期"
          f"（匹配率 {len(rows) / denom * 100:.1f}%）")

    uniformity = check_heat_uniformity(np.array([r["h2"] for r in rows]))
    print(f"H2 均值 {uniformity['mean']:.3f}（n={uniformity['n']}）"
          f" -> {'通过' if uniformity['ok'] else '未通过'}")

    keys = FEATURE_KEYS + [NEGATIVE_CONTROL]
    stats = _feature_stats(rows, keys)
    table = format_feature_table(rows, keys)
    conclusion = format_conclusion(stats, uniformity)
    report = format_report(rows, uniformity, table, conclusion=conclusion)
    print()
    print(report)

    if not args.no_write:
        from pathlib import Path
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"\n报告已写入 {out}")

    sys.exit(0 if uniformity["ok"] else 1)


if __name__ == "__main__":
    main()