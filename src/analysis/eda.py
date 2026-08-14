"""
========================================================================
ETT オイル温度予測 EDA スクリプト (ETT Oil Temperature Forecasting EDA)
========================================================================

仕様書3節で報告されている6つの分析を，指定したETTデータセットに対して
再現する．要約値をコンソールに出力し，グラフを
figures/01_eda/<データセット名>/ に保存する．

■ 対象データセット
・ETTh1, ETTh2: デフォルトの対象（1時間粒度）
・ETTm1, ETTm2: --datasets オプションで指定した場合のみ対象（15分粒度）

■ 出力する図（figures/01_eda/<データセット名>/）
・autocorrelation.png:      3.1 OTの自己相関
・load_lag_correlation.png: 3.2 負荷特徴量とOTのラグ相関
・periodicity.png:          3.3 時刻別・曜日別・月別の周期性
・distribution_shift.png:   3.4 月次OT平均と12/4/4分割の位置
・persistence_baseline.png: 3.6 ホライズン別のPersistence誤差
    ※ 3.5 データ品質は表のみで図はない

■ 実行方法
$ python -m src.analysis.eda --datasets ETTh1 ETTh2
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import plotting
from src.common.dataset import LOAD_COLS, TARGET, load_dataframe, split_bounds, steps_per_hour


# ------------------------------------------------------------------------
# ■ 3.1 OTの自己相関 (Self-autocorrelation of OT)
# ------------------------------------------------------------------------
def section_3_1_autocorrelation(df: pd.DataFrame, sph: int, outdir: Path) -> dict:
    """
    OTが，過去の自分自身の値とどれくらい似ているか（自己相関）を，
    ラグ（時間差）を変えながら計算する．

    ・戻り値: 代表的なラグ（1, 6, 24, 168, 720時間）ごとの自己相関係数
    ・図: autocorrelation.png（ラグ30日分の推移）
    """
    ot = df[TARGET]
    lags_h = [1, 6, 24, 168, 720]
    results = {h: ot.autocorr(lag=h * sph) for h in lags_h}

    max_lag_h = 24 * 30  # 30日分
    lags = np.arange(1, max_lag_h * sph + 1, sph)
    acf_vals = [ot.autocorr(lag=int(l)) for l in lags]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lags / sph, acf_vals)
    ax.set_xlabel("lag (hours)")
    ax.set_ylabel("autocorrelation of OT")
    ax.set_title("3.1 OT self-autocorrelation")
    ax.axhline(0, color="gray", linewidth=0.5)
    plotting.grid(ax)
    plotting.save(fig, outdir / "autocorrelation.png")

    return results


# ------------------------------------------------------------------------
# ■ 3.2 負荷特徴量とOTの相関 (Load-feature correlation with OT)
# ------------------------------------------------------------------------
def section_3_2_load_feature_correlation(df: pd.DataFrame, sph: int, outdir: Path) -> pd.Series:
    """
    6つの負荷特徴量（HUFL等）とOTの相関を計算する．
    さらに，各特徴量をラグ±24時間ずらしながらOTとの相関を再計算し，
    負荷の変化からOTの変化までに時間差（熱応答遅延）があるかを確認する．

    ・戻り値: 各負荷特徴量とOTの相関係数（降順）
    ・図: load_lag_correlation.png（負荷特徴量ごとのラグ相関）
        - 横軸が負のラグ: 負荷特徴量がOTより先行している
        - 横軸が正のラグ: OTが負荷特徴量より先行している
    """
    corr = df[LOAD_COLS].corrwith(df[TARGET]).sort_values(ascending=False)

    max_lag_h = 24
    lag_range = range(-max_lag_h, max_lag_h + 1)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    for ax, col in zip(axes.flat, LOAD_COLS):
        vals = [df[col].shift(l * sph).corr(df[TARGET]) for l in lag_range]
        ax.plot(list(lag_range), vals)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_title(col)
        plotting.grid(ax)
    fig.suptitle("3.2 Lag correlation of load features with OT (negative lag = feature leads OT)")
    fig.supxlabel("lag (hours)")
    plotting.save(fig, outdir / "load_lag_correlation.png")

    return corr


# ------------------------------------------------------------------------
# ■ 3.3 OTの周期性 (Periodicity of OT)
# ------------------------------------------------------------------------
def section_3_3_periodicity(df: pd.DataFrame, outdir: Path) -> dict:
    """
    OTの平均値を「時刻別」「曜日別」「月別」に集計し，どの周期性が
    支配的か（振幅の大小）を比較する．

    ・戻り値: 時刻別・曜日別・月別それぞれの振幅（最大値と最小値の差）
    ・図: periodicity.png（3種類の周期性を並べたグラフ）
    """
    hourly = df.groupby(df.index.hour)[TARGET].mean()
    weekday = df.groupby(df.index.dayofweek)[TARGET].mean()
    monthly = df.groupby(df.index.month)[TARGET].mean()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    hourly.plot(ax=axes[0], marker="o")
    axes[0].set_title(f"OT mean by hour-of-day (amplitude={hourly.max()-hourly.min():.2f})")
    axes[0].set_xlabel("hour")

    weekday.plot(ax=axes[1], marker="o")
    axes[1].set_title(f"OT mean by weekday (amplitude={weekday.max()-weekday.min():.2f})")
    axes[1].set_xlabel("day of week (0=Mon)")

    monthly.plot(ax=axes[2], marker="o")
    axes[2].set_title(f"OT mean by month (amplitude={monthly.max()-monthly.min():.2f})")
    axes[2].set_xlabel("month")

    for ax in axes:
        ax.set_ylabel("mean OT (degC)")
        plotting.grid(ax)
    fig.suptitle("3.3 Periodicity of OT")
    plotting.save(fig, outdir / "periodicity.png")

    return {
        "hourly_amplitude": hourly.max() - hourly.min(),
        "weekday_amplitude": weekday.max() - weekday.min(),
        "monthly_amplitude": monthly.max() - monthly.min(),
    }


# ------------------------------------------------------------------------
# ■ 3.4 分布シフト (Distribution shift across train/val/test)
# ------------------------------------------------------------------------
def section_3_4_distribution_shift(df: pd.DataFrame, sph: int, outdir: Path) -> pd.DataFrame:
    """
    Informer論文の慣例（12/4/4ヶ月，1ヶ月=30日）でtrain/val/testに分割し，
    各期間のOT平均・標準偏差を比較する．期間によってOTの水準が
    大きく異なる場合，「分布シフトがある」と判断する．

    ・分割の境界は common.dataset.split_bounds と共有する．EDAとモデル実験で
      同じ区間を見ていることを保証するため，ここで計算し直さない．
    ・戻り値: train/val/testそれぞれの期間・OT平均・OT標準偏差の一覧
    ・図: distribution_shift.png（月次OT平均の推移とsplit区間の重ね描き）
    """
    bounds = split_bounds(len(df), sph, "informer", df.index)
    splits = {key: df.iloc[lo:hi] for key, (lo, hi) in bounds.items()}

    summary = pd.DataFrame(
        {
            "start": [s.index.min() for s in splits.values()],
            "end": [s.index.max() for s in splits.values()],
            "ot_mean": [s[TARGET].mean() for s in splits.values()],
            "ot_std": [s[TARGET].std() for s in splits.values()],
        },
        index=splits.keys(),
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    monthly_mean = df[TARGET].resample("MS").mean()
    monthly_mean.plot(ax=ax, marker="o")
    for name, s in splits.items():
        ax.axvspan(s.index.min(), s.index.max(), alpha=0.15, label=name)
    ax.set_ylabel("monthly mean OT (degC)")
    ax.set_title("3.4 Monthly mean OT with train/val/test split (distribution shift)")
    plotting.grid(ax)
    ax.legend()
    plotting.save(fig, outdir / "distribution_shift.png")

    return summary


def _max_consecutive_run(s: pd.Series) -> int:
    """同じ値が連続している区間の，最大の長さ（行数）を求める．"""
    same_as_prev = s.eq(s.shift())
    groups = (~same_as_prev).cumsum()
    return int(same_as_prev.groupby(groups).sum().max()) + 1


# ------------------------------------------------------------------------
# ■ 3.5 データ品質チェック (Data quality checks)
# ------------------------------------------------------------------------
def section_3_5_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    各列について，センサー固着・異常値の疑いを機械的にチェックする．
    ここでは図は作らず，表（数値）のみを出力する．

    ・戻り値: 列ごとの「最大連続同一値の長さ」「負値の件数」「ゼロ値の件数」
        - 最大連続同一値が極端に長い: センサー固着（値が変化していない）の疑い
        - OT列の負値: 物理的に油温がマイナスというのは通常考えにくいため要確認
    """
    rows = []
    for col in LOAD_COLS + [TARGET]:
        rows.append(
            {
                "column": col,
                "max_consecutive_identical": _max_consecutive_run(df[col]),
                "n_negative": int((df[col] < 0).sum()),
                "n_zero": int((df[col] == 0).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("column")


# ------------------------------------------------------------------------
# ■ 3.6 Persistenceベースライン (Persistence baseline by horizon)
# ------------------------------------------------------------------------
def section_3_6_persistence_baseline(df: pd.DataFrame, sph: int, outdir: Path) -> pd.Series:
    """
    「直近のOTの値を，そのままホライズン先の予測値として使う」という
    最も単純な予測方法（Persistence）の誤差を，ホライズンごとに計算する．
    以降のモデル構築フェーズで，この誤差を上回れるかどうかが比較の基準となる．

    ・ここでは全期間で算出する．モデルと同じtest区間での値は
      src/train/baseline.py があらためて算出する．
    ・戻り値: ホライズン（1, 6, 24, 96, 336時間）ごとのMAE（平均絶対誤差）
    ・図: persistence_baseline.png（ホライズンとMAEの関係）
    """
    horizons_h = [1, 6, 24, 96, 336]
    mae = {}
    for h in horizons_h:
        steps = h * sph
        pred = df[TARGET].shift(steps)
        mae[h] = (df[TARGET] - pred).abs().mean()
    mae = pd.Series(mae, name="persistence_MAE")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mae.index, mae.to_numpy(), marker="o", **plotting.CONDITION_STYLE["A"])
    plotting.horizon_axis(ax, mae.index)
    ax.set_ylabel("MAE (degC)")
    ax.set_title("3.6 Persistence baseline MAE by horizon (full period)")
    plotting.grid(ax)
    plotting.save(fig, outdir / "persistence_baseline.png")

    return mae


def run_eda(name: str, data_dir: Path, fig_dir: Path) -> None:
    """1つのデータセットに対して，3.1〜3.6の分析をすべて実行し，結果を出力する．"""
    outdir = fig_dir / plotting.FIG_EDA / name
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(name, data_dir)
    sph = steps_per_hour(df)

    print(f"\n{'='*60}\n{name}  (rows={len(df)}, steps/hour={sph})\n{'='*60}")

    print("\n[3.1] OT self-autocorrelation by lag (hours):")
    for h, v in section_3_1_autocorrelation(df, sph, outdir).items():
        print(f"  lag={h:>4}h : corr={v:.3f}")

    print("\n[3.2] Load feature correlation with OT (sorted):")
    print(section_3_2_load_feature_correlation(df, sph, outdir).round(3).to_string())

    print("\n[3.3] Periodicity amplitude (max-min of group means):")
    for k, v in section_3_3_periodicity(df, outdir).items():
        print(f"  {k}: {v:.3f}")

    print("\n[3.4] Distribution shift (12/4/4-month split, Informer convention):")
    print(section_3_4_distribution_shift(df, sph, outdir).round(3).to_string())

    print("\n[3.5] Data quality:")
    print(section_3_5_data_quality(df).to_string())

    print("\n[3.6] Persistence baseline MAE by horizon:")
    print(section_3_6_persistence_baseline(df, sph, outdir).round(3).to_string())

    print(f"\nFigures saved to {outdir}/")


def main():
    """コマンドライン引数を解釈し，指定されたデータセットぶんrun_edaを呼び出す．"""
    parser = argparse.ArgumentParser(description="ETT EDA (spec section 3)")
    parser.add_argument(
        "--datasets", nargs="+", default=["ETTh1", "ETTh2"],
        choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2"],
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--fig-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    for name in args.datasets:
        run_eda(name, args.data_dir, args.fig_dir)


if __name__ == "__main__":
    main()
