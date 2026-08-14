"""
========================================================================
ETT アブレーション実験用データセット構築 (Dataset Builder for Ablation)
========================================================================

仕様書4.0節のアブレーション条件A〜Dに対応する入力・正解ペアを生成する．
12/4/4ヶ月分割・train統計のみによる正規化（4.4節）をここで一元的に扱い，
以降のベースライン・CNN・LightGBMはすべて本モジュール経由で同じ
サンプル集合を参照する．

■ 予測の定式化
・予測実行時刻を t（t以前のデータは既知），ホライズンを h とする．
・入力の時系列窓: [t-N+1, ..., t]  → 仕様書の「過去N時刻」に対応
・予測ターゲット: OT(t+h)          → 仕様書の OT(T)
・条件C/Dの追加入力: 負荷(t+h)     → 仕様書の「時刻Tの負荷」

    ※ 仕様書の入力仕様表は h=1 を前提とした記法（T-N〜T-1）のため，
       h>1 へ一般化した上記の解釈を前提として置く．h=1 のとき両者は一致する．

■ アブレーション条件
・条件B (自己回帰のみ): 窓=[OT]                       aux=なし
・条件C (自己回帰＋負荷): 窓=[OT + 負荷6列]           aux=負荷(t+h)
・条件D (負荷のみ): 窓=[負荷6列]                      aux=負荷(t+h)

    ※ 条件A (Persistence) はモデルを持たないため，OT(t) をそのまま予測値と
       する．比較の公平性のため origin_ot として全条件に同梱する．

■ 動作確認
$ python src/dataset.py --dataset ETTh1 --window 96
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LOAD_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
TARGET = "OT"

# Informer論文の慣例: 1ヶ月=30日として，12/4/4ヶ月でtrain/val/testに分割する．
# （暦月ではなく30日単位のため，実際の境界日は月末とはズレる．）
MONTH_HOURS = 30 * 24
SPLIT_MONTHS = (12, 4, 4)

# 特徴量行列の列順を [OT, 負荷6列] に固定する．以降の列インデックスはこの順序が前提．
FEATURE_COLS = [TARGET] + LOAD_COLS
OT_IDX = 0
LOAD_IDX = list(range(1, 1 + len(LOAD_COLS)))

# 条件ごとの (窓に入れる列, 時刻t+hの補助入力に入れる列)．
CONDITIONS = {
    "B": ([OT_IDX], None),
    "C": ([OT_IDX] + LOAD_IDX, LOAD_IDX),
    "D": (LOAD_IDX, LOAD_IDX),
}


@dataclass
class SplitData:
    """1つの分割（train/val/test）に属するサンプル一式．"""

    x_seq: np.ndarray       # (サンプル数, 窓幅N, チャンネル数) 正規化済みの時系列窓
    x_aux: np.ndarray       # (サンプル数, 6) 正規化済みの時刻t+hの負荷．条件Bではshape=(n, 0)
    y: np.ndarray           # (サンプル数,) 正規化済みのOT(t+h)
    y_raw: np.ndarray       # (サンプル数,) 元スケールのOT(t+h)．評価はこちらで行う
    origin_ot: np.ndarray   # (サンプル数,) 元スケールのOT(t)．条件A（Persistence）の予測値
    target_index: np.ndarray  # (サンプル数,) 元データフレーム上のt+hの行番号

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class Bundle:
    """1つの (データセット, 条件, ホライズン, 窓幅) に対する実験データ一式．"""

    dataset: str
    condition: str
    horizon: int
    window: int
    train: SplitData
    val: SplitData
    test: SplitData
    target_mean: float      # OTの正規化に使ったtrain平均．予測値の逆変換に使う
    target_std: float       # OTの正規化に使ったtrain標準偏差

    @property
    def n_channels(self) -> int:
        """窓入力のチャンネル数（条件Bなら1，Cなら7，Dなら6）．"""
        return self.train.x_seq.shape[2]

    @property
    def n_aux(self) -> int:
        """時刻t+hの補助入力の次元数（条件Bなら0，C・Dなら6）．"""
        return self.train.x_aux.shape[1]

    def inverse_target(self, y_norm: np.ndarray) -> np.ndarray:
        """正規化されたOT予測値を元スケール（摂氏）に戻す．"""
        return y_norm * self.target_std + self.target_mean


def load_dataframe(name: str, data_dir: Path) -> pd.DataFrame:
    """CSVを読み込み，date列をインデックスにした時系列データフレームを返す．"""
    df = pd.read_csv(data_dir / f"{name}.csv", parse_dates=["date"])
    return df.set_index("date").sort_index()


def steps_per_hour(df: pd.DataFrame) -> int:
    """1時間あたりの行数（ETThなら1，ETTmなら4）を，実際の時刻間隔から推定する．"""
    freq_minutes = df.index.to_series().diff().dropna().mode()[0].total_seconds() / 60
    return round(60 / freq_minutes)


def split_bounds(n_rows: int, sph: int) -> dict:
    """
    Informer論文準拠の12/4/4ヶ月分割の境界（行番号）を返す．

    ・戻り値: {"train": (開始, 終了), "val": (...), "test": (...)}（終了は含まない）
    """
    train_months, val_months, test_months = SPLIT_MONTHS
    step = MONTH_HOURS * sph
    train_end = train_months * step
    val_end = (train_months + val_months) * step
    test_end = min((train_months + val_months + test_months) * step, n_rows)
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, test_end),
    }


def _sliding_windows(values: np.ndarray, window: int) -> np.ndarray:
    """
    行方向のスライディング窓を (窓数, 窓幅, 列数) の形で返す．

    ・戻り値のi番目は，元データの行 [i, i+window-1] を覆う窓
      （すなわち予測実行時刻 t = i + window - 1 に対応する）
    """
    view = np.lib.stride_tricks.sliding_window_view(values, window, axis=0)
    return np.ascontiguousarray(view.transpose(0, 2, 1))


def _build_split(
    windows: np.ndarray,
    normed: np.ndarray,
    raw: np.ndarray,
    bounds: tuple,
    window: int,
    horizon: int,
    seq_cols: list,
    aux_cols: list,
) -> SplitData:
    """
    指定した分割区間に属するサンプルを切り出す．

    ・サンプルの所属は「予測ターゲット時刻 t+h がどの区間に入るか」で決める．
      入力窓が前の区間まで遡ることは許容する（推論時点で観測済みのデータを
      使うだけであり，リークではない．Informer系の実装も同じ扱い）．
    """
    lo, hi = bounds

    # 窓の先頭が負にならない最小のターゲット位置: t+h >= h + N - 1
    first_valid = horizon + window - 1
    targets = np.arange(max(lo, first_valid), hi, dtype=np.int64)

    # ターゲット t+h に対応する窓インデックス: t - (N-1) = (t+h) - h - N + 1
    win_idx = targets - horizon - window + 1
    origins = targets - horizon

    x_seq = windows[win_idx][:, :, seq_cols]
    x_aux = (
        normed[targets][:, aux_cols]
        if aux_cols
        else np.zeros((len(targets), 0), dtype=np.float32)
    )
    return SplitData(
        x_seq=np.ascontiguousarray(x_seq),
        x_aux=np.ascontiguousarray(x_aux),
        y=normed[targets, OT_IDX],
        y_raw=raw[targets, OT_IDX],
        origin_ot=raw[origins, OT_IDX],
        target_index=targets,
    )


def build_bundle(
    name: str,
    condition: str,
    horizon_hours: int,
    window: int,
    data_dir: Path,
) -> Bundle:
    """
    1つの (データセット, 条件, ホライズン) に対する train/val/test 一式を構築する．

    ・正規化: train区間の平均・標準偏差のみを使う（4.4節，リーク防止）
    ・horizon_hours: 時間単位で指定し，内部で行数（× steps_per_hour）に換算する
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition} (expected one of {list(CONDITIONS)})")

    df = load_dataframe(name, data_dir)
    sph = steps_per_hour(df)
    raw = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    bounds = split_bounds(len(raw), sph)

    # train区間の統計のみで標準化する．定数列で0除算しないよう標準偏差の下限を置く．
    train_lo, train_hi = bounds["train"]
    mean = raw[train_lo:train_hi].mean(axis=0)
    std = raw[train_lo:train_hi].std(axis=0)
    std[std == 0] = 1.0
    normed = ((raw - mean) / std).astype(np.float32)

    horizon = horizon_hours * sph
    windows = _sliding_windows(normed, window)
    seq_cols, aux_cols = CONDITIONS[condition]

    splits = {
        key: _build_split(
            windows, normed, raw, bounds[key], window, horizon, seq_cols, aux_cols
        )
        for key in ("train", "val", "test")
    }
    return Bundle(
        dataset=name,
        condition=condition,
        horizon=horizon_hours,
        window=window,
        target_mean=float(mean[OT_IDX]),
        target_std=float(std[OT_IDX]),
        **splits,
    )


def main() -> None:
    """動作確認用: 指定した条件・ホライズンでの形状とサンプル数を表示する．"""
    parser = argparse.ArgumentParser(description="ETTアブレーション用データセットの構築確認")
    parser.add_argument("--dataset", default="ETTh1", choices=["ETTh1", "ETTh2"])
    parser.add_argument("--window", type=int, default=96, help="入力窓幅N（時刻数）")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 24, 96, 336])
    parser.add_argument("--conditions", nargs="+", default=["B", "C", "D"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    df = load_dataframe(args.dataset, args.data_dir)
    sph = steps_per_hour(df)
    bounds = split_bounds(len(df), sph)
    print(f"=== {args.dataset} (rows={len(df)}, steps/hour={sph}, window={args.window}) ===")
    for key, (lo, hi) in bounds.items():
        print(f"  {key:5s}: rows[{lo}:{hi}]  {df.index[lo]} 〜 {df.index[hi - 1]}")

    for cond in args.conditions:
        for h in args.horizons:
            b = build_bundle(args.dataset, cond, h, args.window, args.data_dir)
            print(
                f"  cond={cond} h={h:4d}h  channels={b.n_channels} aux={b.n_aux}  "
                f"train={len(b.train):5d} val={len(b.val):5d} test={len(b.test):5d}  "
                f"x_seq={b.train.x_seq.shape}"
            )


if __name__ == "__main__":
    main()
