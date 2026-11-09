"""测试 LSTM 序列对齐契约：特征到第 D 期 → 预测第 D+1 期"""
import numpy as np
from ssq.models.lstm_model import build_lstm_sequences


def test_lstm_sequence_ends_at_label_row():
    """序列最后一行的特征期应等于标签行：序列以行 i 结尾、以 y[i] 为标签"""
    n, f, seq_len = 30, 4, 5
    X_feat = np.arange(n * f, dtype=np.float32).reshape(n, f)
    y_red = np.zeros((n, 33), dtype=np.float32)
    y_red[:, 0] = np.arange(n)  # 标签值编码行号，便于断言配对
    y_blue = np.zeros((n, 16), dtype=np.float32)

    X_seq, y_r, y_b = build_lstm_sequences(X_feat, y_red, y_blue, seq_len)

    assert len(X_seq) == n - seq_len + 1
    # 第一条序列：最后一行 = X_feat[seq_len-1]，标签 = y_red[seq_len-1]
    np.testing.assert_array_equal(X_seq[0][-1], X_feat[seq_len - 1])
    assert y_r[0][0] == seq_len - 1
    # 最后一条序列：最后一行 = X_feat[n-1]，标签 = y_red[n-1]
    np.testing.assert_array_equal(X_seq[-1][-1], X_feat[n - 1])
    assert y_r[-1][0] == n - 1
