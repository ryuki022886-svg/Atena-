"""
========================================================================
条件A: Persistenceベースライン (Persistence Baseline)
========================================================================

仕様書4.0節の条件A（比較の基準線）を，以降のCNN実験とまったく同じ
サンプル集合の上で正式に算出する．

■ 条件Aの定義
・予測値 = OT(t)（予測実行時刻の値をそのまま将来へ横引きする．学習なし）
・正解値 = OT(t+h)

    ※ 仕様書3.6節では全期間で算出したが，CNNとの比較には同一区間での値が
       必要なため，本スクリプトでは12/4/4分割のtest区間で算出し直す．
       参考として全期間の値も併記する．

■ 出力
・results/baseline.csv: データセット×ホライズン別のMAE・RMSE
・figures/baseline_persistence.png: ホライズンに対する誤差の推移

■ 実行方法
$ python src/baseline.py --datasets ETTh1 ETTh2 --window 96
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset import TARGET, build_bundle, load_dataframe, split_bounds, steps_per_hour
from metrics import score


def evaluate_persistence(name: str, horizon: int, window: int, data_dir: Path) -> dict:
    """
    1つの (データセット, ホライズン) について条件Aを評価する．

    ・条件Bのバンドルを借用するが，使うのは origin_ot と y_raw のみ．
      条件によらず同じ値なので，どの条件を借りても結果は変わらない．
    """
    bundle = build_bundle(name, "B", horizon, window, data_dir)

    row = {"dataset": name, "horizon": horizon, "window": window}
    for split_name in ("val", "test"):
        split = getattr(bundle, split_name)
        for key, value in score(split.y_raw, split.origin_ot).items():
            row[f"{split_name}_{key}"] = value
    row["test_n"] = len(bundle.test)
    return row


def full_period_mae(name: str, horizon: int, data_dir: Path) -> float:
    """仕様書3.6節と同じ「全期間」でのMAE．表の整合性確認用に併記する．"""
    ot = load_dataframe(name, data_dir)[TARGET].to_numpy()
    sph = steps_per_hour(load_dataframe(name, data_dir))
    step = horizon * sph
    return float(np.mean(np.abs(ot[step:] - ot[:-step])))


def plot_baseline(table: pd.DataFrame, fig_path: Path) -> None:
    """ホライズンに対する誤差の伸び方を，データセット別に描画する．"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for metric, ax in zip(("mae", "rmse"), axes):
        for name, group in table.groupby("dataset"):
            group = group.sort_values("horizon")
            ax.plot(group["horizon"], group[f"test_{metric}"], marker="o", label=name)
        ax.set_xscale("log")
        ax.set_xticks(sorted(table["horizon"].unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("horizon (hours)")
        ax.set_ylabel(f"test {metric.upper()} (degC)")
        ax.set_title(f"Persistence baseline: {metric.upper()}")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="条件A（Persistence）ベースラインの算出")
    parser.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2"])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 24, 96, 336])
    parser.add_argument("--window", type=int, default=96, help="入力窓幅N（サンプル集合を揃えるために使用）")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--result-dir", type=Path, default=Path("results"))
    parser.add_argument("--fig-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        evaluate_persistence(name, h, args.window, args.data_dir)
        for name in args.datasets
        for h in args.horizons
    ]
    table = pd.DataFrame(rows)

    csv_path = args.result_dir / "baseline.csv"
    table.to_csv(csv_path, index=False)
    fig_path = args.fig_dir / "baseline_persistence.png"
    plot_baseline(table, fig_path)

    for name in args.datasets:
        df = load_dataframe(name, args.data_dir)
        sph = steps_per_hour(df)
        bounds = split_bounds(len(df), sph)
        lo, hi = bounds["test"]
        print(f"\n=== {name} 条件A (Persistence) ===")
        print(f"  test区間: {df.index[lo]} 〜 {df.index[hi - 1]}  ({hi - lo}時刻)")
        print(f"  test区間のOT: 平均{df[TARGET].iloc[lo:hi].mean():.2f} "
              f"標準偏差{df[TARGET].iloc[lo:hi].std():.2f}")
        print(f"  {'horizon':>8} {'test MAE':>9} {'test RMSE':>10} {'val MAE':>9} {'全期間MAE':>10}")
        for h in args.horizons:
            r = table[(table.dataset == name) & (table.horizon == h)].iloc[0]
            full = full_period_mae(name, h, args.data_dir)
            print(f"  {h:7d}h {r.test_mae:9.2f} {r.test_rmse:10.2f} "
                  f"{r.val_mae:9.2f} {full:10.2f}")

    print(f"\n保存先: {csv_path}, {fig_path}")


if __name__ == "__main__":
    main()
