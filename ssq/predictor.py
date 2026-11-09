import numpy as np
from .config import RED_MAX, BLUE_MAX, RED_COUNT
from .features import build_feature_matrix, build_next_features
from .models.baseline import BaselineModel
from .models.xgb_model import XGBPredictor
from .models.lstm_model import LSTMPredictor, build_lstm_input
from .models.ensemble import EnsembleModel
from . import db

# 缓存键 v2：区别于对齐修复前的旧口径缓存（旧模型特征/标签错位，不得复用）
XGB_CACHE_KEY = "xgb_v2"
LSTM_CACHE_KEY = "lstm_v2"


class Predictor:
    """预测编排器：加载数据 → 训练 → 预测 → 推荐"""

    def __init__(self, draws: list[dict], weights: dict[str, float] = None):
        self.draws = draws
        self.baseline = BaselineModel()
        self.xgb = XGBPredictor()
        self.lstm = LSTMPredictor()
        self.ensemble = EnsembleModel(weights)
        self._trained = False
        self._feat_cache = {}  # 特征缓存

    def _get_features(self):
        """获取特征矩阵（带缓存）"""
        if self._feat_cache.get("draws") is not self.draws:
            X, y_red, y_blue = build_feature_matrix(self.draws)
            self._feat_cache = {"draws": self.draws, "X": X, "y_red": y_red, "y_blue": y_blue}
        c = self._feat_cache
        return c["X"], c["y_red"], c["y_blue"]

    def load_or_train(self):
        """尝试加载缓存模型，否则重新训练"""
        latest = self.draws[-1]["draw_no"]

        # 尝试加载XGBoost
        cached_xgb = db.load_model(XGB_CACHE_KEY)
        cached_lstm = db.load_model(LSTM_CACHE_KEY)

        if cached_xgb and cached_xgb[0] == latest:
            self.xgb.deserialize(cached_xgb[1])
            self.baseline.fit(self.draws)
            if cached_lstm and cached_lstm[0] == latest:
                self.lstm.deserialize(cached_lstm[1])
            # 构建特征缓存供 predict 使用
            self._get_features()
            self._trained = True
            return

        self.train()

    def train(self):
        """训练所有模型"""
        print("  构建特征矩阵...")
        X, y_red, y_blue = self._get_features()

        # 生产模型用全部已知数据训练（回测评估另用扣除 holdout 的临时模型，见 _backtest）
        X_train, y_red_train, y_blue_train = X, y_red, y_blue
        test_size = min(50, len(X) // 5)
        X_test = X[-test_size:]
        y_red_test = y_red[-test_size:]
        y_blue_test = y_blue[-test_size:]

        print("  训练基线模型...")
        self.baseline.fit(self.draws)

        print("  训练XGBoost模型（33+16个分类器）...")
        self.xgb.fit(X_train, y_red_train, y_blue_train)

        print("  训练LSTM模型（特征序列）...")
        self.lstm.fit(X_train, y_red_train, y_blue_train)

        # 缓存模型
        latest = self.draws[-1]["draw_no"]
        db.save_model(XGB_CACHE_KEY, latest, self.xgb.serialize())
        if self.lstm.model is not None:
            db.save_model(LSTM_CACHE_KEY, latest, self.lstm.serialize())

        # 回测评估
        self._backtest(X_test, y_red_test, y_blue_test,
                       self.draws[:len(self.draws) - test_size])

        self._trained = True
        print("  模型训练完成")

    def _backtest(self, X_test: np.ndarray, y_red_test: np.ndarray,
                  y_blue_test: np.ndarray, train_draws: list[dict]):
        """在测试集上评估模型表现（用集成权重，含 LSTM）

        注意：所有模型都只在训练集上拟合，避免数据泄漏
        """
        print("\n  === 回测评估（最近{}期）===".format(len(X_test)))

        X_all, y_red_all, y_blue_all = self._get_features()
        split_idx = len(X_all) - len(X_test)
        X_train_bt = X_all[:split_idx]
        y_red_train_bt = y_red_all[:split_idx]
        y_blue_train_bt = y_blue_all[:split_idx]

        # 所有临时模型只在训练集上拟合（避免数据泄漏）
        eval_baseline = BaselineModel()
        eval_baseline.fit(train_draws)

        eval_xgb = XGBPredictor()
        eval_xgb.fit(X_train_bt, y_red_train_bt, y_blue_train_bt)

        eval_lstm = LSTMPredictor(seq_len=self.lstm.seq_len, hidden=self.lstm.hidden)
        eval_lstm.fit(X_train_bt, y_red_train_bt, y_blue_train_bt)

        red_hits = []
        blue_hits = []

        for i in range(len(X_test)):
            predictions = {}

            red_base, blue_base = eval_baseline.predict_proba()
            predictions["baseline"] = (red_base, blue_base)

            red_xgb, blue_xgb = eval_xgb.predict_proba(X_test[i:i+1])
            predictions["xgboost"] = (red_xgb, blue_xgb)

            # LSTM 用测试点之前的特征序列（序列末端=测试点特征行，与标签对齐）
            if eval_lstm.model is not None and eval_lstm._available:
                feat_start = split_idx + i
                seq_start = max(0, feat_start - self.lstm.seq_len + 1)
                feat_seq = X_all[seq_start:feat_start + 1]
                red_lstm, blue_lstm = eval_lstm.predict_proba(feat_seq)
                predictions["lstm"] = (red_lstm, blue_lstm)

            red_comb, blue_comb = self.ensemble.combine(predictions)

            top6_red = set(np.argsort(red_comb)[-6:])
            actual_red = set(np.where(y_red_test[i] == 1)[0])
            red_hits.append(len(top6_red & actual_red))

            pred_blue = np.argmax(blue_comb)
            actual_blue = np.where(y_blue_test[i] == 1)[0]
            blue_hits.append(1 if pred_blue in actual_blue else 0)

        red_hits = np.array(red_hits)
        blue_hits = np.array(blue_hits)

        print(f"  红球命中分布: 0球={np.sum(red_hits==0)}期, 1球={np.sum(red_hits==1)}期, "
              f"2球={np.sum(red_hits==2)}期, 3球+={np.sum(red_hits>=3)}期")
        print(f"  红球平均命中: {red_hits.mean():.2f}/6 (理论期望: {6*6/33:.2f})")
        print(f"  蓝球命中率: {blue_hits.mean()*100:.1f}% (理论期望: {100/16:.1f}%)")
        print()

    def _collect_predictions(self, X_next: np.ndarray) -> dict:
        """收集各模型预测概率；LSTM 不可用（未训练/torch缺失）时排除，不让 uniform 稀释融合"""
        predictions = {}

        red_base, blue_base = self.baseline.predict_proba()
        predictions["baseline"] = (red_base, blue_base)

        if len(getattr(self.xgb, "red_models", [])) > 0:
            red_xgb, blue_xgb = self.xgb.predict_proba(X_next)
            predictions["xgboost"] = (red_xgb, blue_xgb)

        if getattr(self.lstm, "model", None) is not None and self.lstm._available:
            # LSTM 输入序列以最新一期特征结尾，与预测目标对齐
            X, _, _ = self._get_features()
            red_lstm, blue_lstm = self.lstm.predict_proba(build_lstm_input(X, self.draws, self.lstm.seq_len))
            predictions["lstm"] = (red_lstm, blue_lstm)

        return predictions

    def predict(self, n: int = 3) -> list[dict]:
        """生成n组推荐号码"""
        if not self._trained:
            raise RuntimeError("模型未训练，请先调用 train() 或 load_or_train()")

        # 下一期特征
        X_next = build_next_features(self.draws)

        # 获取各模型预测并融合
        predictions = self._collect_predictions(X_next)
        red_final, blue_final = self.ensemble.combine(predictions)

        # 生成推荐
        return self.ensemble.generate_recommendations(red_final, blue_final, n_sets=n)

    def tune(self, n_trials: int = 50) -> dict:
        """用 Optuna 搜索 XGBoost 最优超参数"""
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        X, y_red, y_blue = self._get_features()
        test_size = min(50, len(X) // 5)
        X_train, X_val = X[:-test_size], X[-test_size:]
        y_red_train, y_red_val = y_red[:-test_size], y_red[-test_size:]
        y_blue_train, y_blue_val = y_blue[:-test_size], y_blue[-test_size:]

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }
            from xgboost import XGBClassifier
            red_pos_weight = (RED_MAX - 6) / 6
            blue_pos_weight = (BLUE_MAX - 1) / 1

            # 用前5个红球+前4个蓝球快速评估
            score = 0
            for i in range(5):
                m = XGBClassifier(**params, eval_metric="logloss", verbosity=0,
                                  n_jobs=-1, scale_pos_weight=red_pos_weight)
                split = max(1, int(len(X_train) * 0.9))
                m.fit(X_train[:split], y_red_train[:split, i],
                      eval_set=[(X_train[split:], y_red_train[split:, i])],
                      verbose=False, early_stopping_rounds=20)
                score += m.score(X_val, y_red_val[:, i])
            for i in range(4):
                m = XGBClassifier(**params, eval_metric="logloss", verbosity=0,
                                  n_jobs=-1, scale_pos_weight=blue_pos_weight)
                split = max(1, int(len(X_train) * 0.9))
                m.fit(X_train[:split], y_blue_train[:split, i],
                      eval_set=[(X_train[split:], y_blue_train[split:, i])],
                      verbose=False, early_stopping_rounds=20)
                score += m.score(X_val, y_blue_val[:, i])
            return score / 9

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_params
