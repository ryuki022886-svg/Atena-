"""
========================================================================
ETT アブレーション実験用データセット構築 (Dataset Builder for Ablation)
========================================================================

仕様書4.0節のアブレーション条件A〜Dに対応する入力・正解ペアを生成する．
分割とtrain統計のみによる正規化（4.4節）をここで一元的に扱い，
以降のベースライン・CNNはすべて本モジュール経由で同じサンプル集合を参照する．

■ 分割の切り替え (split_mode)
・informer: 仕様書4.4節準拠の12/4/4ヶ月分割．ベンチマークと比較できるが
  test区間が冬季4ヶ月に限られ，季節に偏った評価になる．
・fiscal: 年度1（2016-07〜2017-06）で学習し，年度2（2017-07〜2018-06）を
  まるごとテストする．test区間が12ヶ月あり全季節をカバーする．
  年度1の末尾2ヶ月をvalに充てる．

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

■ 予測対象の切り替え (target_mode)
・absolute: OT(t+h) を直接予測する．仕様書どおりの素直な定式化．
・delta: OT(t+h) - OT(t) を予測する．窓・補助入力・ターゲットのすべてから
  予測実行時点tの値を引くため，水準がサンプルごとに0へ揃い，経年ドリフトの
  影響を受けない．

    ※ 経年ドリフトはETTh1では油温(-0.90σ)に，ETTh2では負荷(最大-1.47σ)に現れる．
       入る経路は違うが両データセットに存在するため，全チャンネルに適用する．

■ deltaモードにおける条件Dの扱い
・条件Dはモデル入力にOTを持たないが，deltaモードでも実行できる．
  起点OT(t)はモデルの外側で予測値に足し戻すためだけに使うので，
  「モデルは負荷しか見ていない」という条件Dの性質は保たれる．
・この形にすると4条件が同じ土俵に乗る．deltaモードでは条件A(Persistence)が
  「常にΔ=0と答えるモデル」に相当するため，D vs A が
  「負荷情報は『変化しない』という仮定を上回るか」という直接比較になる．

■ 動作確認
$ python -m src.common.dataset --dataset ETTh1 --window 96 --target-mode delta
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

# 年度分割（fiscalモード）の設定．データが2016-07始まりのため7月を年度の開始とする．
FISCAL_SPLIT = pd.Timestamp("2017-07-01")
FISCAL_VAL_MONTHS = 2

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
    split_mode: str         # "informer": 12/4/4ヶ月分割 / "fiscal": 年度1で学習し年度2でテスト
    target_mode: str        # "absolute": OT(t+h)を直接予測 / "delta": OT(t+h)-OT(t)を予測
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

    def inverse_target(self, y_norm: np.ndarray, origin_ot: np.ndarray) -> np.ndarray:
        """
        モデル出力を元スケール（摂氏）のOT予測値に戻す．

        ・absoluteモード: train統計で逆変換するだけ
        ・deltaモード: 予測しているのは変化量なので，起点のOT(t)に足し戻す
        ・origin_ot: 各サンプルの予測実行時点の実測OT．absoluteモードでは使わない
        """
        if self.target_mode == "delta":
            return origin_ot + y_norm * self.target_std
        return y_norm * self.target_std + self.target_mean


def load_dataframe(name: str, data_dir: Path) -> pd.DataFrame:
    """CSVを読み込み，date列をインデックスにした時系列データフレームを返す．"""
    df = pd.read_csv(data_dir / f"{name}.csv", parse_dates=["date"])
    return df.set_index("date").sort_index()


def steps_per_hour(df: pd.DataFrame) -> int:
    """1時間あたりの行数（ETThなら1，ETTmなら4）を，実際の時刻間隔から推定する．"""
    freq_minutes = df.index.to_series().diff().dropna().mode()[0].total_seconds() / 60
    return round(60 / freq_minutes)


def split_bounds(n_rows: int, sph: int, mode: str = "informer", index=None) -> dict:
    """
    train/val/testの境界（行番号）を返す．

    ・戻り値: {"train": (開始, 終了), "val": (...), "test": (...)}（終了は含まない）
    ・mode="informer": 仕様書4.4節準拠の12/4/4ヶ月分割（1ヶ月=30日）．
      ベンチマークとの比較可能性があるが，test区間が冬季4ヶ月に限られる．
    ・mode="fiscal": 年度1で学習し年度2をまるごとテストする．
      test区間が12ヶ月あり全季節をカバーするため，季節に依存しない評価ができる．
      「1年運用してデータを貯め，翌年から予測を使う」という実運用の姿にも対応する．
        - index（DatetimeIndex）が必要．年度境界は実際の日付で判定する．
    """
    if mode == "informer":
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

    if mode == "fiscal":
        if index is None:
            raise ValueError("fiscalモードにはDatetimeIndexが必要")
        # 年度1と年度2の境界．ここより前が年度1，以降が年度2．
        year2_start = int(np.searchsorted(index.to_numpy(), FISCAL_SPLIT.to_datetime64()))
        # 年度1の末尾2ヶ月をvalに充て，残りをtrainとする．
        val_start = year2_start - FISCAL_VAL_MONTHS * MONTH_HOURS * sph
        if val_start <= 0:
            raise ValueError("年度1が短すぎてvalを確保できない")
        return {
            "train": (0, val_start),
            "val": (val_start, year2_start),
            "test": (year2_start, n_rows),
        }

    raise ValueError(f"unknown split mode: {mode} (expected 'informer' or 'fiscal')")


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
    target_mode: str,
) -> SplitData:
    """
    指定した分割区間に属するサンプルを切り出す．

    ・サンプルの所属は「予測ターゲット時刻 t+h がどの区間に入るか」で決める．
      入力窓が前の区間まで遡ることは許容する（推論時点で観測済みのデータを
      使うだけであり，リークではない．Informer系の実装も同じ扱い）．
    ・deltaモードでは，窓・補助入力・ターゲットのすべてから「予測実行時点t の値」を
      引く．これによりサンプルごとに水準が0に揃い，経年ドリフトの影響を受けなくなる．
      スケールはtrain統計の標準偏差で共通に割るため，サンプル間の大小関係は保たれる．
    """
    lo, hi = bounds

    # 窓の先頭が負にならない最小のターゲット位置: t+h >= h + N - 1
    first_valid = horizon + window - 1
    targets = np.arange(max(lo, first_valid), hi, dtype=np.int64)

    # ターゲット t+h に対応する窓インデックス: t - (N-1) = (t+h) - h - N + 1
    win_idx = targets - horizon - window + 1
    origins = targets - horizon

    seq = windows[win_idx]
    aux = normed[targets]
    y = normed[targets, OT_IDX]
    if target_mode == "delta":
        # normed は (raw - train平均) / train標準偏差 なので，正規化値どうしの差は
        # (raw - raw起点) / train標準偏差 に等しい．平均項は差し引きで消える．
        anchor = normed[origins]
        seq = seq - anchor[:, None, :]
        aux = aux - anchor
        y = y - anchor[:, OT_IDX]

    x_aux = (
        aux[:, aux_cols]
        if aux_cols
        else np.zeros((len(targets), 0), dtype=np.float32)
    )
    return SplitData(
        x_seq=np.ascontiguousarray(seq[:, :, seq_cols]),
        x_aux=np.ascontiguousarray(x_aux),
        y=y,
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
    target_mode: str = "absolute",
    split_mode: str = "informer",
) -> Bundle:
    """
    1つの (データセット, 条件, ホライズン) に対する train/val/test 一式を構築する．

    ・正規化: train区間の平均・標準偏差のみを使う（4.4節，リーク防止）
    ・horizon_hours: 時間単位で指定し，内部で行数（× steps_per_hour）に換算する
    ・target_mode: "absolute" はOT(t+h)を直接予測する．"delta" はOT(t+h)-OT(t)を
      予測し，経年ドリフトの影響を受けないようにする．
        - 条件Dもdeltaモードで扱える．起点OT(t)はモデルの外側で足し戻すためだけに
          使うので，「モデルは負荷しか見ていない」という性質は保たれる（冒頭の解説を参照）．
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition} (expected one of {list(CONDITIONS)})")
    if target_mode not in ("absolute", "delta"):
        raise ValueError(f"unknown target_mode: {target_mode}")

    df = load_dataframe(name, data_dir)
    sph = steps_per_hour(df)
    raw = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    bounds = split_bounds(len(raw), sph, split_mode, df.index)

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
            windows, normed, raw, bounds[key], window, horizon, seq_cols, aux_cols, target_mode
        )
        for key in ("train", "val", "test")
    }
    return Bundle(
        dataset=name,
        condition=condition,
        horizon=horizon_hours,
        window=window,
        split_mode=split_mode,
        target_mode=target_mode,
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
    parser.add_argument("--target-mode", default="absolute", choices=["absolute", "delta"])
    parser.add_argument("--split-mode", default="informer", choices=["informer", "fiscal"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    df = load_dataframe(args.dataset, args.data_dir)
    sph = steps_per_hour(df)
    bounds = split_bounds(len(df), sph, args.split_mode, df.index)
    print(f"=== {args.dataset} (rows={len(df)}, steps/hour={sph}, window={args.window}, "
          f"split={args.split_mode}) ===")
    for key, (lo, hi) in bounds.items():
        print(f"  {key:5s}: rows[{lo}:{hi}]  {df.index[lo]} 〜 {df.index[hi - 1]}")

    for cond in args.conditions:
        for h in args.horizons:
            b = build_bundle(args.dataset, cond, h, args.window, args.data_dir,
                             args.target_mode, args.split_mode)
            print(
                f"  cond={cond} h={h:4d}h  channels={b.n_channels} aux={b.n_aux}  "
                f"train={len(b.train):5d} val={len(b.val):5d} test={len(b.test):5d}  "
                f"x_seq={b.train.x_seq.shape}"
            )


if __name__ == "__main__":
    main()
