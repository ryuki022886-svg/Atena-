"""
========================================================================
経年トレンドの可視化 (Year-over-Year Trend Visualization)
========================================================================

仕様書3.4節の分布シフトについて，その正体が「季節性では説明できない
経年変化」であることを可視化する．モデル実験でETTh1が失敗しETTh2が
成功した理由を，データ側から説明するための図を生成する．

■ 年度の定義
・データ期間が2016-07〜2018-06のため，7月始まりで年度を定義する．
    - 年度1: 2016-07-01 〜 2017-06-30
    - 年度2: 2017-07-01 〜 2018-06-30
・こうすると2つの年度が同じ季節構成になり，同月どうしを直接比較できる．

■ 出力
・figures/02_trend/timeseries.png: OTの全期間推移と各分割の位置
・figures/02_trend/year_over_year.png: 同じ月の年度間比較
・results/trend_year_over_year.csv: 同月比較の数値

■ 実行方法
$ python -m src.analysis.trend
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import plotting
from src.common.dataset import FISCAL_SPLIT, TARGET, load_dataframe, split_bounds, steps_per_hour

# 7月始まりで並べた月の順序．季節の流れどおりに読めるようにする．
MONTH_ORDER = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]


def fiscal_year(index: pd.DatetimeIndex) -> np.ndarray:
    """各時刻がどちらの年度に属するかを1/2で返す．"""
    return np.where(index < FISCAL_SPLIT, 1, 2)


def year_over_year_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    同じ月について年度1と年度2の平均OTを並べ，その差を求める．

    ・季節性を揃えた上で比較するため，差はそのまま経年変化の大きさを表す．
    """
    grouped = df[TARGET].groupby([fiscal_year(df.index), df.index.month]).mean()
    table = grouped.unstack(0)
    table.columns = ["year1", "year2"]
    table = table.reindex(MONTH_ORDER).dropna()
    table["diff"] = table["year2"] - table["year1"]
    table.index.name = "month"
    return table


def plot_timeseries(frames: dict, outpath: Path) -> Path:
    """
    OTの全期間推移を描き，12/4/4分割と年度分割の位置を重ねる．

    ・時間粒度のままでは密すぎるため，日次平均を主線とし生データを薄く重ねる．
    """
    fig, axes = plt.subplots(len(frames), 1, figsize=(12, 3.4 * len(frames)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, (name, df) in zip(axes, frames.items()):
        ot = df[TARGET]
        ax.plot(ot.index, ot.to_numpy(), color="lightsteelblue", linewidth=0.4, label="hourly")
        daily = ot.resample("1D").mean()
        ax.plot(daily.index, daily.to_numpy(), color="tab:blue", linewidth=1.2, label="daily mean")

        # 仕様書準拠の12/4/4分割の位置を帯で示す．
        bounds = split_bounds(len(df), steps_per_hour(df))
        colors = {"train": "tab:green", "val": "tab:orange", "test": "tab:red"}
        for key, (lo, hi) in bounds.items():
            ax.axvspan(df.index[lo], df.index[hi - 1], color=colors[key], alpha=0.08)
            ax.text(df.index[(lo + hi) // 2], ot.max(), key,
                    ha="center", va="top", fontsize=8, color=colors[key])

        # 年度の境界．左が年度1，右が年度2．
        ax.axvline(FISCAL_SPLIT, color="black", linestyle="--", linewidth=1.0)
        ax.text(FISCAL_SPLIT, ot.min(), " year1 | year2", fontsize=8, va="bottom")

        ax.set_ylabel("OT (degC)")
        ax.set_title(f"{name}: oil temperature over the full period")
        plotting.grid(ax)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("date")
    return plotting.save(fig, outpath)


def plot_year_over_year(tables: dict, outpath: Path) -> Path:
    """同じ月の年度間比較を，折れ線と差分の棒グラフで描く．"""
    fig, axes = plt.subplots(2, len(tables), figsize=(6 * len(tables), 7), squeeze=False)
    positions = np.arange(len(MONTH_ORDER))
    labels = [f"{m}" for m in MONTH_ORDER]

    for col, (name, table) in enumerate(tables.items()):
        top, bottom = axes[0][col], axes[1][col]

        top.plot(positions, table["year1"], marker="o", label="year1 (2016-07 to 2017-06)")
        top.plot(positions, table["year2"], marker="s", label="year2 (2017-07 to 2018-06)")
        top.set_xticks(positions)
        top.set_xticklabels(labels)
        top.set_xlabel("month (fiscal order, starting July)")
        top.set_ylabel("mean OT (degC)")
        top.set_title(f"{name}: same month, different year")
        plotting.grid(top)
        top.legend()

        colors = ["tab:red" if d < 0 else "tab:blue" for d in table["diff"]]
        bottom.bar(positions, table["diff"], color=colors)
        bottom.axhline(0, color="black", linewidth=0.8)
        bottom.axhline(table["diff"].mean(), color="gray", linestyle="--", linewidth=1.0)
        bottom.text(0, table["diff"].mean(), f" mean {table['diff'].mean():+.2f} degC",
                    fontsize=8, va="bottom", color="gray")
        bottom.set_xticks(positions)
        bottom.set_xticklabels(labels)
        bottom.set_xlabel("month (fiscal order, starting July)")
        bottom.set_ylabel("year2 - year1 (degC)")
        bottom.set_title(f"{name}: year-over-year change")
        plotting.grid(bottom)

    return plotting.save(fig, outpath)


def main() -> None:
    parser = argparse.ArgumentParser(description="経年トレンドの可視化")
    parser.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--result-dir", type=Path, default=Path("results"))
    parser.add_argument("--fig-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    frames = {name: load_dataframe(name, args.data_dir) for name in args.datasets}
    tables = {name: year_over_year_table(df) for name, df in frames.items()}

    for name, table in tables.items():
        print(f"=== {name} 同月の年度間比較（℃）===")
        print(table.round(2).to_string())
        print(f"  平均差: {table['diff'].mean():+.2f}℃  "
              f"全月で低下: {'はい' if (table['diff'] < 0).all() else 'いいえ'}\n")

    combined = pd.concat(tables, names=["dataset"]).reset_index()
    csv_path = args.result_dir / "trend_year_over_year.csv"
    combined.to_csv(csv_path, index=False)

    outdir = args.fig_dir / plotting.FIG_TREND
    paths = [
        plot_timeseries(frames, outdir / "timeseries.png"),
        plot_year_over_year(tables, outdir / "year_over_year.png"),
    ]
    print(f"保存先: {', '.join(str(p) for p in paths)}, {csv_path}")


if __name__ == "__main__":
    main()
