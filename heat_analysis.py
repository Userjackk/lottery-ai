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
    FEATURE_KEYS, bh_correct, bootstrap_ci, build_dataset,
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


def format_feature_table(rows: list[dict], feature_keys: list[str],
                         n_groups: int = 3) -> str:
    """特征三档 log(H2) 均值 + CI + 多元回归 β/p（BH 校正）"""
    usable = [r for r in rows
              if not np.isnan(float(r.get("h2", np.nan)))
              and (r.get("sales") or 0) > 0]
    if not usable:
        return "（无可用于分析的行）"

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

    lines = [
        "| 特征 | 低档 logH2 | 中档 logH2 | 高档 logH2 | 高档-低档 | 95% CI | β | p(BH) | 判定 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    none = "—"
    for i, k in enumerate(feature_keys):
        vals = np.array([float(r.get(k, 0.0)) for r in usable])
        if np.std(vals) == 0:
            lines.append(f"| {k} | {none} | {none} | {none} | {none} | {none} | {none} | {none} | 无方差 |")
            continue

        g = quantile_groups(vals, n_groups)
        means = []
        for gi in range(n_groups):
            sub = y[g == gi]
            means.append(f"{sub.mean():+.3f}" if len(sub) else none)

        lo_v, hi_v = y[g == 0], y[g == n_groups - 1]
        # 极度偏斜的特征会出现空分档（如 95% 取 0），此时不报档间差
        if len(lo_v) >= 2 and len(hi_v) >= 2:
            diff = float(hi_v.mean() - lo_v.mean())
            ci_lo, ci_hi = _bootstrap_diff_ci(hi_v, lo_v)
            diff_s = f"{diff:+.3f}"
            ci_s = f"[{ci_lo:+.3f}, {ci_hi:+.3f}]"
        else:
            diff_s = ci_s = none

        beta = float(fit["beta"][2 + i])
        pv = float(p_adj[i])
        if np.isnan(pv):
            judge = "无方差"
        elif pv < 0.05 and beta > 0:
            judge = "显著(正)"
        elif pv < 0.05 and beta < 0:
            judge = "显著(负)"
        else:
            judge = "不显著"

        lines.append(
            f"| {k} | {means[0]} | {means[1]} | {means[2]} | {diff_s} | "
            f"{ci_s} | {beta:+.4f} | {pv:.4f} | {judge} |"
        )

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

    lines.append("")
    lines.append("### 交叉核对（单变量 vs 多变量，控制变量相同）")
    lines.append("")
    lines.append("| 特征 | 单变量 β | 单变量 p(BH) | 多变量 β | 一致性 |")
    lines.append("|---|---|---|---|---|")
    for i, k in enumerate(feature_keys):
        mb = float(fit["beta"][2 + i])
        if np.isnan(uni_beta[i]):
            lines.append(f"| {k} | {none} | {none} | {mb:+.4f} | 无方差 |")
            continue
        ub, up = uni_beta[i], float(uni_adj[i])
        if ub * mb < 0:
            agree = "相反(疑共线)"
        else:
            agree = "一致"
        lines.append(f"| {k} | {ub:+.4f} | {up:.4g} | {mb:+.4f} | {agree} |")

    return "\n".join(lines)


def format_report(rows: list[dict], uniformity: dict, feature_table: str,
                  top_n: int = 20) -> str:
    lines = ["# 双色球号码热度实证分析", ""]

    lines.append("## 均匀性自检")
    lines.append(f"- 有效样本：{uniformity['n']} 期")
    lines.append(f"- H2 均值：{uniformity['mean']:.3f}（应为 1）")
    verdict = "通过" if uniformity["ok"] else "未通过 —— 口径可能有误，下面的结论仅供参考"
    lines.append(f"- 判定：{verdict}")
    lines.append("")

    lines.append("## 特征检验")
    lines.append("")
    lines.append(feature_table)
    lines.append("")

    lines.append(f"## 最热 {top_n} 期（按 H2）")
    lines.append("")
    lines.append("| 期号 | H2 | H1 | 一等奖注数 | 二等奖注数 |")
    lines.append("|---|---|---|---|---|")
    top = sorted(rows, key=lambda r: -(r.get("h2") if r.get("h2") is not None else -1))[:top_n]
    for r in top:
        lines.append(
            f"| {r['draw_no']} | {r['h2']:.2f} | {r['h1']:.2f} | {r['n1']} | {r['n2']} |"
        )
    lines.append("")

    lines.append("## 结论")
    sig = feature_table.count("显著(正)") + feature_table.count("显著(负)")
    if sig == 0:
        lines.append("没有任何特征通过显著性检验，不应据此改动避号规则。")
    else:
        lines.append(f"有 {sig} 个特征显著，详见上表；方向以 β 符号为准。")
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
        rows = fetch_with_prizes()
        print(f"  抓取 {len(rows)} 期，写入 {insert_prize_stats(rows)} 期")

    draws = db.get_all_draws()
    prize = get_all_prize_stats()
    rows = build_dataset(prize, draws)

    denom = min(len(draws), len(prize)) or 1
    print(f"开奖 {len(draws)} 期 / 奖级 {len(prize)} 期 / 匹配 {len(rows)} 期"
          f"（匹配率 {len(rows) / denom * 100:.1f}%）")

    uniformity = check_heat_uniformity(np.array([r["h2"] for r in rows]))
    print(f"H2 均值 {uniformity['mean']:.3f}（n={uniformity['n']}）"
          f" → {'通过' if uniformity['ok'] else '未通过'}")

    table = format_feature_table(rows, FEATURE_KEYS)
    report = format_report(rows, uniformity, table)
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