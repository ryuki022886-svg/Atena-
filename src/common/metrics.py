"""
========================================================================
評価指標 (Evaluation Metrics)
========================================================================

仕様書4.4節で指定された評価指標を定義する．ベースラインもCNNも本モジュールを
共有し，指標の実装差による比較のブレを防ぐ．

■ 前提
・入力は元スケール（摂氏）の値とする．正規化された値のまま計算しないこと．
"""
import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均絶対誤差．外れ値の影響が小さく，誤差の大きさを直感的に読める．"""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """二乗平均平方根誤差．大きな外し方をより強く罰する．"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAEとRMSEをまとめて返す．"""
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred)}
