import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ..config import RED_MAX, BLUE_MAX
from ..features import build_next_features


if HAS_TORCH:
    class _LSTMNet(nn.Module):
        def __init__(self, input_dim=203, hidden=64, red_out=RED_MAX, blue_out=BLUE_MAX):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, num_layers=2, batch_first=True, dropout=0.2)
            self.red_head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, red_out), nn.Sigmoid())
            self.blue_head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, blue_out), nn.Sigmoid())

        def forward(self, x):
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.red_head(last), self.blue_head(last)


def build_lstm_sequences(X_feat: np.ndarray, y_red: np.ndarray, y_blue: np.ndarray,
                         seq_len: int):
    """构建 LSTM 序列：序列以行 i 结尾、以 y[i] 为标签（特征到 D 期 → 预测 D+1 期）

    Args:
        X_feat: (n_samples, n_features) 特征矩阵，每行对应一期
        y_red / y_blue: (n_samples, ...) 每行的标签
        seq_len: 序列长度

    Returns:
        X_seq: (n_seq, seq_len, n_features)
        y_red / y_blue: (n_seq, ...)
    """
    n_samples = len(X_feat)
    X_seq, y_r, y_b = [], [], []
    for i in range(seq_len - 1, n_samples):
        X_seq.append(X_feat[i - seq_len + 1:i + 1])
        y_r.append(y_red[i])
        y_b.append(y_blue[i])
    return np.array(X_seq, dtype=np.float32), np.array(y_r), np.array(y_b)


def build_lstm_input(X_feat: np.ndarray, draws: list[dict], seq_len: int) -> np.ndarray:
    """构建预测用的 LSTM 输入序列：最近 seq_len 行特征 + 下一期特征结尾

    序列 = [..., X_feat[n-1], X_next]，其中 X_feat[n-1] 是最新一期，
    X_next 是 build_next_features 的输出，保证预测目标与序列末端对齐。
    """
    next_feat = build_next_features(draws)
    rows = list(X_feat[-(seq_len - 1):]) + [next_feat[0]]
    return np.array(rows, dtype=np.float32)


class LSTMPredictor:
    """LSTM序列预测器（基于特征序列）"""

    def __init__(self, seq_len: int = 20, hidden: int = 64, epochs: int = 100, batch_size: int = 64):
        self.seq_len = seq_len
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self._available = HAS_TORCH

    def _build_feature_sequences(self, X_feat: np.ndarray):
        """从特征矩阵构建 LSTM 输入序列和标签

        Args:
            X_feat: (n_samples, n_features) 特征矩阵，每行对应一期

        Returns:
            X_seq: (n_seq, seq_len, n_features) 输入序列
            y_red: (n_seq, 33) 红球标签
            y_blue: (n_seq, 16) 蓝球标签
        """
        n_samples, n_feat = X_feat.shape
        X_seq, y_red, y_blue = [], [], []

        for i in range(self.seq_len, n_samples - 1):
            X_seq.append(X_feat[i - self.seq_len:i])
            # 标签：下一期的 one-hot（从特征矩阵无法直接获取，需要额外信息）
            # 这里用特征向量的前49维近似（遗漏值=0 表示该号码当期出现）
            # 但这不准确，所以标签从原始 draws 提取更好

        return np.array(X_seq, dtype=np.float32)

    def fit(self, X_feat: np.ndarray, y_red: np.ndarray, y_blue: np.ndarray):
        """用特征矩阵训练 LSTM

        Args:
            X_feat: (n_samples, n_features) 特征矩阵
            y_red: (n_samples, 33) 红球标签
            y_blue: (n_samples, 16) 蓝球标签
        """
        if not self._available:
            return
        n_samples = len(X_feat)
        if n_samples < self.seq_len + 50:
            return

        # 构建序列：seq_len 个连续特征向量（到行 i 结束）→ 预测第 i 期标签
        X_seq, y_r, y_b = build_lstm_sequences(X_feat, y_red, y_blue, self.seq_len)

        X_t = torch.tensor(np.array(X_seq), dtype=torch.float32)
        y_red_t = torch.tensor(np.array(y_r), dtype=torch.float32)
        y_blue_t = torch.tensor(np.array(y_b), dtype=torch.float32)

        # 80/20 训练/验证分割
        split = max(1, int(len(X_t) * 0.8))
        train_ds = TensorDataset(X_t[:split], y_red_t[:split], y_blue_t[:split])
        val_X, val_red, val_blue = X_t[split:], y_red_t[split:], y_blue_t[split:]
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        n_feat = X_feat.shape[1]
        self.model = _LSTMNet(input_dim=n_feat, hidden=self.hidden)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5)
        criterion = nn.BCELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            # 训练
            self.model.train()
            train_loss = 0
            for xb, rb, bb in train_loader:
                red_pred, blue_pred = self.model(xb)
                loss = criterion(red_pred, rb) + criterion(blue_pred, bb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # 验证
            self.model.eval()
            with torch.no_grad():
                red_val, blue_val = self.model(val_X)
                val_loss = (criterion(red_val, val_red) + criterion(blue_val, val_blue)).item()

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    break

        # 恢复最佳模型
        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict_proba(self, X_feat_recent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """用最近的特征向量序列预测

        Args:
            X_feat_recent: (>=seq_len, n_features) 最近若干期的特征向量
        """
        if not self._available or self.model is None:
            return np.ones(RED_MAX) / RED_MAX, np.ones(BLUE_MAX) / BLUE_MAX

        X = torch.tensor(X_feat_recent[-self.seq_len:].reshape(1, self.seq_len, -1), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            red_p, blue_p = self.model(X)
        return red_p.numpy().flatten(), blue_p.numpy().flatten()

    def serialize(self) -> bytes:
        import io
        buf = io.BytesIO()
        torch.save(self.model.state_dict(), buf)
        return buf.getvalue()

    def deserialize(self, data: bytes):
        if not self._available:
            return
        import io
        # 从序列化数据推断 input_dim
        buf = io.BytesIO(data)
        state = torch.load(buf, weights_only=True)
        input_dim = state["lstm.weight_ih_l0"].shape[1]
        self.model = _LSTMNet(input_dim=input_dim, hidden=self.hidden)
        self.model.load_state_dict(state)
