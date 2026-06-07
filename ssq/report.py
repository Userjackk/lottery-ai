from . import analyzer


def format_analysis(draws: list[dict], window: int = 30) -> str:
    """生成中文统计分析报告"""
    if not draws:
        return "暂无数据，请先运行: python main.py init"

    latest = draws[-1]
    lines = [
        "=" * 45,
        "  双色球历史数据分析报告",
        "=" * 45,
        f"数据范围: 第{draws[0]['draw_no']}期 ~ 第{latest['draw_no']}期 (共{len(draws)}期)",
        f"分析窗口: 最近{min(window, len(draws))}期",
        "",
    ]

    # 冷热号
    hc = analyzer.hot_cold_analysis(draws)
    lines.append("【冷热号分析】")
    lines.append(f"  热号 ({len(hc['hot'])}个): {_fmt_nums(hc['hot'])}")
    lines.append(f"  温号 ({len(hc['warm'])}个): {_fmt_nums(hc['warm'])}")
    lines.append(f"  冷号 ({len(hc['cold'])}个): {_fmt_nums(hc['cold'])}")
    lines.append("")

    # 遗漏值
    omit = analyzer.omission_analysis(draws)
    red_omits = sorted(omit["red"].items(), key=lambda x: x[1], reverse=True)
    lines.append("【遗漏值TOP10】")
    top10 = red_omits[:10]
    for num, o in top10:
        bar = "█" * min(o // 5, 20)
        lines.append(f"  {num:02d}: {o:3d}期 {bar}")
    lines.append("")

    # 奇偶/大小
    pat = analyzer.pattern_analysis(draws, window)
    lines.append("【奇偶/大小比分布】")
    lines.append(f"  奇偶比: {pat['odd_even_dist']}")
    lines.append(f"  大小比: {pat['big_small_dist']}")
    lines.append("")

    # 和值
    sa = analyzer.sum_analysis(draws, window)
    lines.append("【红球和值】")
    lines.append(f"  范围: {sa['min']} ~ {sa['max']}，均值: {sa['avg']:.1f}")
    lines.append(f"  近5期: {sa['recent']}")
    lines.append("")

    # 连号
    ca = analyzer.consecutive_analysis(draws, window)
    lines.append("【连号分析】")
    lines.append(f"  连号出现率: {ca['frequency']:.0%}")
    lines.append(f"  平均连号对数: {ca['avg_pairs']:.1f}")
    lines.append("")

    # 近期开奖
    lines.append("【最近5期开奖】")
    for d in draws[-5:]:
        reds = " ".join(f"{d[f'red{i}']:02d}" for i in range(1, 7))
        lines.append(f"  第{d['draw_no']}期: {reds} + {d['blue']:02d}")

    return "\n".join(lines)


def format_history(draws: list[dict], count: int = 10) -> str:
    """格式化历史开奖记录"""
    if not draws:
        return "暂无数据"
    recent = draws[-count:]
    lines = [f"最近 {len(recent)} 期开奖记录:", "-" * 50]
    for d in reversed(recent):
        reds = " ".join(f"{d[f'red{i}']:02d}" for i in range(1, 7))
        lines.append(f"  第{d['draw_no']}期 ({d['date']}): {reds} + {d['blue']:02d}")
    return "\n".join(lines)


def format_status(total: int, latest: str | None) -> str:
    lines = [
        "【项目状态】",
        f"  数据总量: {total} 期",
        f"  最新期号: {latest or '无'}",
    ]
    return "\n".join(lines)


def format_prediction_single(pred: dict, draw_no: str = "") -> str:
    """格式化单注推荐"""
    if not pred:
        return "预测失败，请先训练模型: python main.py train"
    reds = " ".join(f"{n:02d}" for n in sorted(pred["red"]))
    blue = f"{pred['blue']:02d}"
    title = f"第{draw_no}期" if draw_no else ""
    return f"双色球{title}: 红球 {reds} 蓝球 {blue}"


def format_predictions(predictions: list[dict], draw_no: str = "") -> str:
    """格式化预测推荐结果"""
    if not predictions:
        return "预测失败，请先训练模型: python main.py train"

    style_names = {0: "稳健型", 1: "均衡型", 2: "搏冷型"}
    style_reasons = {
        0: "高频热号为主 + 遗漏回补",
        1: "热冷搭配 + 区间均衡",
        2: "遗漏回补 + 逆势搏冷",
    }

    lines = [
        "=" * 45,
        f"  双色球{'第' + draw_no + '期 ' if draw_no else ''}预测推荐",
        "=" * 45,
        "",
    ]

    for i, p in enumerate(predictions):
        style = style_names.get(i, f"方案{i + 1}")
        reason = style_reasons.get(i, "")
        conf = p.get("confidence", 0)
        stars = "*" * round(conf / 20) + "." * (5 - round(conf / 20))
        reds = " ".join(f"{n:02d}" for n in sorted(p["red"]))
        blue = f"{p['blue']:02d}"

        lines.append(f"【方案{chr(65 + i)} · {style}】 综合置信度: {stars} ({conf:.0f}%)")
        lines.append(f"  红球: {reds}")
        lines.append(f"  蓝球: {blue}")
        if reason:
            lines.append(f"  理由: {reason}")
        lines.append("")

    lines.append("  [!] 彩票为随机事件，预测仅供参考娱乐")
    lines.append("=" * 45)
    return "\n".join(lines)


def _fmt_nums(nums: list[int]) -> str:
    return " ".join(f"{n:02d}" for n in sorted(nums)) if nums else "无"
