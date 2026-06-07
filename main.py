import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import click
from ssq.db import init_db, insert_draws, get_all_draws, get_draw_count, get_latest_draw_number, get_next_draw_number
from ssq.scraper import fetch_all, fetch_latest
from ssq import report as rpt


@click.group()
def cli():
    """双色球AI预测工具"""
    pass


@cli.command()
def init():
    """初始化：建库 + 抓取全部历史数据"""
    click.echo("正在初始化数据库...")
    init_db()
    click.echo("正在抓取历史开奖数据（首次可能需要几分钟）...")
    draws = fetch_all()
    count = insert_draws(draws)
    total = get_draw_count()
    click.echo(f"完成！新增 {count} 期，共 {total} 期数据")


@cli.command()
def update():
    """增量更新最新开奖数据"""
    init_db()
    click.echo("正在检查最新数据...")
    draws = fetch_latest()
    count = insert_draws(draws)
    total = get_draw_count()
    latest = get_latest_draw_number()
    if count:
        click.echo(f"新增 {count} 期，当前共 {total} 期（最新: {latest}）")
    else:
        click.echo(f"数据已是最新（截至第 {latest} 期，共 {total} 期）")


@cli.command()
@click.option("--last", default=30, help="分析窗口大小（期数）")
def analyze(last):
    """分析历史数据（冷热号、遗漏、趋势）"""
    init_db()
    draws = get_all_draws()
    click.echo(rpt.format_analysis(draws, window=last))


@cli.command()
@click.option("--last", default=10, help="显示最近N期")
def history(last):
    """查看历史开奖记录"""
    init_db()
    draws = get_all_draws()
    click.echo(rpt.format_history(draws, count=last))


@cli.command()
def status():
    """查看数据和模型状态"""
    init_db()
    total = get_draw_count()
    latest = get_latest_draw_number()
    click.echo(rpt.format_status(total, latest))


@cli.command()
@click.option("--count", default=1, help="生成N组推荐（默认1组）")
@click.option("--no-train", is_flag=True, help="跳过自动训练")
@click.option("--weights", default=None, help="模型权重，格式: baseline=0.2,xgboost=0.6,lstm=0.2")
def predict(count, no_train, weights):
    """生成预测推荐号码"""
    init_db()
    # 自动检查并抓取缺失的开奖数据
    click.echo("正在检查数据更新...")
    new_draws = fetch_latest()
    new_count = insert_draws(new_draws)
    latest_after = get_latest_draw_number()
    if new_count:
        click.echo(f"新增 {new_count} 期（最新: {latest_after}）")
    else:
        click.echo(f"数据已是最新（第 {latest_after} 期）")

    draws = get_all_draws()
    if len(draws) < 100:
        click.echo("数据不足（需至少100期），请先运行: python main.py init")
        return

    # 解析权重
    w = None
    if weights:
        w = {}
        for pair in weights.split(","):
            k, v = pair.strip().split("=")
            w[k.strip()] = float(v)

    from ssq.predictor import Predictor
    if not no_train:
        click.echo("正在加载/训练模型...")
    p = Predictor(draws, weights=w)
    p.load_or_train()
    preds = p.predict(n=count)
    next_no = get_next_draw_number()
    if count == 1:
        click.echo(rpt.format_prediction_single(preds[0], next_no))
    else:
        click.echo(rpt.format_predictions(preds, next_no))


@cli.command()
def train():
    """训练/重训预测模型"""
    init_db()
    draws = get_all_draws()
    if len(draws) < 100:
        click.echo("数据不足（需至少100期），请先运行: python main.py init")
        return
    from ssq.predictor import Predictor
    click.echo("正在训练模型...")
    p = Predictor(draws)
    p.train()
    click.echo("模型训练完成")


@cli.command()
@click.option("--trials", default=50, help="搜索轮数（默认50）")
def tune(trials):
    """用 Optuna 搜索 XGBoost 最优超参数"""
    init_db()
    draws = get_all_draws()
    if len(draws) < 100:
        click.echo("数据不足（需至少100期），请先运行: python main.py init")
        return
    try:
        import optuna  # noqa: F401
    except ImportError:
        click.echo("请先安装 optuna: pip install optuna")
        return
    from ssq.predictor import Predictor
    click.echo(f"正在搜索最优超参数（{trials}轮）...")
    p = Predictor(draws)
    best = p.tune(n_trials=trials)
    click.echo(f"\n最优参数: {best}")
    click.echo("正在用最优参数重训模型...")
    from xgboost import XGBClassifier
    # 用最优参数重新训练
    p.xgb = type(p.xgb)()
    # 注入最优参数到 XGBPredictor
    p.xgb._best_params = best
    p.train()
    click.echo("超参优化完成")


if __name__ == "__main__":
    cli()
