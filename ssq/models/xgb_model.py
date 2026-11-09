import joblib
import numpy as np
from xgboost import XGBClassifier
from ..config import RED_MAX, BLUE_MAX


class XGBPredictor:
    """XGBoost 二分类预测器"""

    def __init__(self):
        self.red_models: list[XGBClassifier] = []
        self.blue_models: list[XGBClassifier] = []
        self._best_params: dict = None  # Optuna 超参搜索结果

    def fit(self, X: np.ndarray, y_red: np.ndarray, y_blue: np.ndarray):
        """训练33个红球+16个蓝球二分类器（并行）"""
        # 类别不平衡权重：红球出现率约6/33，蓝球约1/16
        red_pos_weight = (RED_MAX - 6) / 6   # 约4.5
        blue_pos_weight = (BLUE_MAX - 1) / 1  # 约15

        # 用最后10%做early stopping验证集
        split = max(1, int(len(X) * 0.9))
        X_train, X_val = X[:split], X[split:]

        # 超参：优先用 Optuna 搜索结果，否则默认值
        # 注：降低 max_depth、提高正则化，减少对近期频率的过拟合
        base_params = self._best_params or {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.03,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
        }

        def _train_one(i, y_all, pos_weight, is_val_valid):
            if is_val_valid[i]:
                model = XGBClassifier(
                    **base_params,
                    eval_metric="logloss",
                    verbosity=0,
                    n_jobs=-1,
                    scale_pos_weight=pos_weight,
                    early_stopping_rounds=20,
                )
                model.fit(X_train, y_all[:split, i], eval_set=[(X_val, y_all[split:, i])], verbose=False)
            else:
                # 验证集缺正/负样本时无法早停，去掉 early_stopping_rounds 否则 XGBoost 报错
                model = XGBClassifier(
                    **base_params,
                    eval_metric="logloss",
                    verbosity=0,
                    n_jobs=-1,
                    scale_pos_weight=pos_weight,
                )
                model.fit(X_train, y_all[:split, i], verbose=False)
            return model

        # 预计算验证集是否有效（正负样本都存在）
        red_val_valid = [(y_red[split:, i].sum() > 0 and (1 - y_red[split:, i]).sum() > 0) for i in range(RED_MAX)]
        blue_val_valid = [(y_blue[split:, i].sum() > 0 and (1 - y_blue[split:, i]).sum() > 0) for i in range(BLUE_MAX)]

        # 并行训练：XGBoost 内部已用 n_jobs，模型间用 threading 避免进程 fork 开销
        from joblib import Parallel, delayed
        self.red_models = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_train_one)(i, y_red, red_pos_weight, red_val_valid) for i in range(RED_MAX)
        )
        self.blue_models = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_train_one)(i, y_blue, blue_pos_weight, blue_val_valid) for i in range(BLUE_MAX)
        )

    def predict_proba(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回红球(33,)概率和蓝球(16,)概率"""
        red_probs = np.array([m.predict_proba(X)[:, 1] for m in self.red_models]).flatten()
        blue_probs = np.array([m.predict_proba(X)[:, 1] for m in self.blue_models]).flatten()
        return red_probs, blue_probs

    def serialize(self) -> bytes:
        import io
        buf = io.BytesIO()
        joblib.dump({"red": self.red_models, "blue": self.blue_models}, buf)
        return buf.getvalue()

    def deserialize(self, data: bytes):
        import io
        buf = io.BytesIO(data)
        models = joblib.load(buf)
        self.red_models = models["red"]
        self.blue_models = models["blue"]
