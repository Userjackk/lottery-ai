# AICP - 双色球AI预测工具

基于机器学习的双色球号码预测工具，自动采集历史数据、统计分析、多模型融合预测。

## 快速开始

```bash
git clone https://github.com/Userjackk/lottery-ai.git
cd lottery-ai
pip install -r requirements.txt

# 首次使用（抓取历史数据，约需1分钟）
python main.py init

# 预测下期号码（自动检查更新）
python main.py predict
```

输出：`双色球第2026060期: 红球 14 23 24 26 28 32 蓝球 04`

## 命令一览

| 命令 | 说明 | 示例 |
|------|------|------|
| `init` | 首次初始化，抓取全部历史数据 | `python main.py init` |
| `update` | 增量更新最新开奖 | `python main.py update` |
| `predict` | 预测下期号码（自动更新+自动训练） | `python main.py predict` |
| `predict --count 5` | 生成5组推荐 | `python main.py predict --count 5` |
| `predict --no-train` | 跳过自动训练，用缓存模型预测 | `python main.py predict --no-train` |
| `train` | 强制重训模型 | `python main.py train` |
| `analyze` | 统计分析（冷热号/遗漏/趋势） | `python main.py analyze` |
| `analyze --last 50` | 分析最近50期 | `python main.py analyze --last 50` |
| `history` | 查看历史开奖 | `python main.py history` |
| `history --last 20` | 最近20期 | `python main.py history --last 20` |
| `status` | 查看数据和模型状态 | `python main.py status` |

## 日常使用流程

```
python main.py predict
```

这一条命令会自动完成：检查数据更新 → 加载/训练模型 → 输出推荐号码。

## 工作原理

### 数据

- **来源**：中国福彩官网 API（cwl.gov.cn）
- **范围**：2013年至今，约2000+期
- **存储**：SQLite 数据库 `data/ssq.db`

### 特征工程

对每期开奖提取 203 维特征向量：

| 特征组 | 维度 | 内容 |
|--------|------|------|
| 红球1-33 | 132 | 每个号码的遗漏值、10期频率、30期频率、50期频率 |
| 蓝球1-16 | 64 | 每个号码的遗漏值、10期频率、30期频率、50期频率 |
| 全局特征 | 7 | 奇偶比、大小比、三区分布、和值、连号 |

### 模型架构

三个模型加权融合：

| 模型 | 权重 | 原理 |
|------|------|------|
| XGBoost | 60% | 49个二分类器（33红+16蓝），每个输出 P(该号码出现) |
| 基线模型 | 25% | 频率 × 遗漏因子加权采样 |
| LSTM | 15% | 最近20期序列趋势预测 |

### 预测流程

```
历史开奖数据
    │
    ▼
特征提取 (203维)
    │
    ├──→ XGBoost (33个概率) ──→ 加权 ──→ 取TOP6 ──→ 红球推荐
    ├──→ 基线模型 (33个概率) ──→ 融合
    └──→ LSTM    (33个概率) ──→ 起来

蓝球同理，取TOP1
```

核心思路：不是预测"哪6个号会中"，而是对每个号码独立计算出现概率，取最高的6个。将组合问题转化为独立二分类。

### 推荐风格

`--count > 1` 时生成三种风格：

- **稳健型**：直接取概率最高的6个红球
- **均衡型**：TOP4 + 加权随机2个（热冷搭配）
- **搏冷型**：侧重遗漏值高的号码（逆势回补）

## 项目结构

```
AICP/
├── main.py                  # CLI 入口
├── requirements.txt         # 依赖
├── ssq/
│   ├── config.py            # 常量、路径
│   ├── db.py                # SQLite 数据库操作
│   ├── scraper.py           # 数据采集（福彩官网API）
│   ├── analyzer.py          # 统计分析
│   ├── features.py          # 特征工程
│   ├── predictor.py         # 预测编排器
│   ├── report.py            # 中文报告输出
│   └── models/
│       ├── baseline.py      # 频率加权基线
│       ├── xgb_model.py     # XGBoost 二分类
│       ├── lstm_model.py    # LSTM 序列预测
│       └── ensemble.py      # 多模型融合
└── data/
    └── ssq.db               # 数据库（运行时创建）
```

## 依赖

- Python 3.10+
- scikit-learn, xgboost, pandas, numpy
- requests, beautifulsoup4, lxml
- click, joblib
- torch（可选，仅 LSTM 需要）

```bash
pip install -r requirements.txt
```

## 重要说明

彩票开奖是独立随机事件，任何模型都无法真正预测结果。本工具仅用于：
- 数据采集和统计分析
- 机器学习流水线实践
- 娱乐参考，请理性购彩
