"""
========================================================================
実験結果の集計と可視化 (Result Aggregation and Visualization)
========================================================================

複数の定式化（分割方法 × 予測対象）で実行したアブレーション結果を1つの表に
まとめ，報告スライドに載せる図を生成する．

■ 集計する軸
・条件: A(Persistence) / B(自己回帰) / C(自己回帰＋負荷) / D(負荷のみ)
・ホライズン: 1h / 24h / 96h / 336h
・データセット: ETTh1 / ETTh2
・定式化: absolute（OT(t+h)を直接予測）/ delta（OT(t+h)-OT(t)を予測）

■ 中心的な指標
・Persistence比 = 条件XのMAE / 条件AのMAE
    - 1.0未満なら「何もしないモデル」に勝っている．絶対値のMAEはデータセットや
      期間で大きく変わるため，比で見ると条件間の優劣が読みやすい．

■ 出力
・results/summary.csv: 全結果を1行=1設定でまとめた表
・figures/summary_mae.png: ホライズン別MAEの推移（定式化ごとに並べる）
・figures/summary_ratio.png: Persistence比のヒートマップ
・figures/summary_timeseries.png: 予測と実測の重ね書き（代表例）

■ 実行方法
$ python src/evaluate.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset import load_dataframe

CONDITION_ORDER = ["A", "B", "C", "D"]
CONDITION_LABEL = {
    "A": "A: Persistence",
    "B": "B: OT lag only",
    "C": "C: OT lag + load",
    "D": "D: load only",
}


def collect_results(result_dir: Path) -> pd.DataFrame:
    """
    results/ 配下の ablation_*.csv とベースラインを1つの表に統合する．

    ・条件Aは学習を伴わないため baseline.csv から取り込み，全定式化に複製する．
      定式化によらず「直近値をそのまま出す」予測は同じなので，同一値で問題ない．
    """
    frames = []
    for path in sorted(result_dir.glob("ablation_*.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit(f"結果が見つかりません: {result_dir}/ablation_*.csv")
    table = pd.concat(frames, ignore_index=True)

    baselines = pd.concat(
        [pd.read_csv(p) for p in sorted(result_dir.glob("baseline_*.csv"))], ignore_index=True
    )
    baselines = baselines[["dataset", "horizon", "split_mode", "test_mae", "test_rmse", "val_mae"]]
    baselines = baselines.assign(condition="A")

    # 条件Aは学習を伴わないので target_mode に依存しない．存在する定式化ぶん複製する．
    rows = [table]
    for target_mode in table["target_mode"].unique():
        rows.append(baselines.assign(target_mode=target_mode))
    merged = pd.concat(rows, ignore_index=True)
    merged = merged[merged.set_index(["split_mode", "target_mode"]).index.isin(
        table.set_index(["split_mode", "target_mode"]).index.unique())]
    return merged.sort_values(["split_mode", "target_mode", "dataset", "condition", "horizon"])


def add_ratio(table: pd.DataFrame) -> pd.DataFrame:
    """
    各行に「同じ設定でのPersistence比」を付ける．1.0未満ならベースラインに勝ち．

    ・キーが重複する（同一設定に4条件ぶんの行がある）ため，インデックス同士の
      除算ではなく明示的なmergeで対応付ける．
    """
    key = ["split_mode", "target_mode", "dataset", "horizon"]
    base = (table[table["condition"] == "A"][key + ["test_mae"]]
            .rename(columns={"test_mae": "baseline_mae"}))
    merged = table.merge(base, on=key, how="left", validate="many_to_one")
    return merged.assign(persistence_ratio=merged["test_mae"] / merged["baseline_mae"])


def plot_mae(table: pd.DataFrame, fig_path: Path) -> None:
    """ホライズンに対するMAEの推移を，(定式化 × データセット)のパネルで並べる．"""
    combos = sorted(table.groupby(["split_mode", "target_mode"]).groups.keys())
    datasets = sorted(table["dataset"].unique())
    fig, axes = plt.subplots(len(combos), len(datasets),
                             figsize=(5.5 * len(datasets), 3.8 * len(combos)), squeeze=False)

    for row, (split_mode, target_mode) in enumerate(combos):
        for col, dataset in enumerate(datasets):
            ax = axes[row][col]
            subset = table[(table["split_mode"] == split_mode)
                           & (table["target_mode"] == target_mode)
                           & (table["dataset"] == dataset)]
            for cond in CONDITION_ORDER:
                group = subset[subset["condition"] == cond].sort_values("horizon")
                if group.empty:
                    continue
                style = {"linestyle": "--", "color": "black"} if cond == "A" else {}
                ax.plot(group["horizon"], group["test_mae"], marker="o",
                        label=CONDITION_LABEL[cond], **style)
            ax.set_xscale("log")
            ax.set_xticks(sorted(subset["horizon"].unique()))
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.set_xlabel("horizon (hours)")
            ax.set_ylabel("test MAE (degC)")
            ax.set_title(f"{dataset} / {split_mode} / {target_mode}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)


def plot_ratio_heatmap(table: pd.DataFrame, fig_path: Path) -> None:
    """
    Persistence比をヒートマップで示す．

    ・1.0を境に色を分け，ベースラインに勝ったかどうかを一目で分かるようにする．
    """
    subset = table[table["condition"] != "A"]
    combos = sorted(subset.groupby(["split_mode", "target_mode"]).groups.keys())
    fig, axes = plt.subplots(1, len(combos), figsize=(5.2 * len(combos), 4.2), squeeze=False)

    for ax, (split_mode, target_mode) in zip(axes[0], combos):
        part = subset[(subset["split_mode"] == split_mode)
                      & (subset["target_mode"] == target_mode)]
        pivot = part.pivot_table(index=["dataset", "condition"],
                                 columns="horizon", values="persistence_ratio")
        data = pivot.to_numpy()
        # 1.0を中央にした発散カラーマップ．青=勝ち，赤=負け．
        im = ax.imshow(data, cmap="RdBu_r", vmin=0.0, vmax=2.0, aspect="auto")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels([f"{h}h" for h in pivot.columns])
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels([f"{d} {c}" for d, c in pivot.index], fontsize=8)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(f"{split_mode} / {target_mode}\n(ratio to Persistence, <1 is better)",
                     fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)


def plot_timeseries(result_dir: Path, data_dir: Path, fig_path: Path,
                    pred_tag: str, dataset: str, horizon: int, days: int) -> None:
    """
    予測と実測の重ね書き．数値表では伝わらない「どう外しているか」を示す．

    ・test区間の先頭から指定日数ぶんを切り出して描画する．
    """
    pred_dir = result_dir / f"preds_{pred_tag}"
    index = load_dataframe(dataset, data_dir).index

    conditions = [c for c in ("B", "C", "D")
                  if (pred_dir / f"{dataset}_{c}_h{horizon}_s42.npz").exists()]
    if not conditions:
        return

    fig, ax = plt.subplots(figsize=(12, 4.2))
    span = days * 24
    first = np.load(pred_dir / f"{dataset}_{conditions[0]}_h{horizon}_s42.npz")
    times = index[first["target_index"][:span]]

    ax.plot(times, first["true"][:span], color="black", linewidth=1.6, label="actual")
    ax.plot(times, first["persistence"][:span], color="gray", linestyle="--",
            linewidth=1.0, label="A: Persistence")
    for cond in conditions:
        data = np.load(pred_dir / f"{dataset}_{cond}_h{horizon}_s42.npz")
        ax.plot(times, data["pred"][:span], linewidth=1.0, label=CONDITION_LABEL[cond])

    ax.set_xlabel("date")
    ax.set_ylabel("OT (degC)")
    ax.set_title(f"{dataset} horizon={horizon}h ({pred_tag}): prediction vs actual, first {days} days of test")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="アブレーション結果の集計と可視化")
    parser.add_argument("--result-dir", type=Path, default=Path("results"))
    parser.add_argument("--fig-dir", type=Path, default=Path("figures"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--ts-tag", default="informer_delta", help="時系列図に使う予測ディレクトリの識別子")
    parser.add_argument("--ts-dataset", default="ETTh1")
    parser.add_argument("--ts-horizon", type=int, default=24)
    parser.add_argument("--ts-days", type=int, default=21)
    args = parser.parse_args()

    table = add_ratio(collect_results(args.result_dir))
    table.to_csv(args.result_dir / "summary.csv", index=False)

    for (split_mode, target_mode), part in table.groupby(["split_mode", "target_mode"]):
        print(f"\n=== split={split_mode} / target={target_mode} : test MAE（摂氏）===")
        pivot = part.pivot_table(index=["dataset", "horizon"], columns="condition", values="test_mae")
        print(pivot.reindex(columns=[c for c in CONDITION_ORDER if c in pivot.columns]).round(3).to_string())
        print(f"--- Persistence比（<1.0で勝ち）---")
        ratio = part[part["condition"] != "A"].pivot_table(
            index=["dataset", "horizon"], columns="condition", values="persistence_ratio")
        print(ratio.round(2).to_string())

    plot_mae(table, args.fig_dir / "summary_mae.png")
    plot_ratio_heatmap(table, args.fig_dir / "summary_ratio.png")
    plot_timeseries(args.result_dir, args.data_dir, args.fig_dir / "summary_timeseries.png",
                    args.ts_tag, args.ts_dataset, args.ts_horizon, args.ts_days)
    print(f"\n保存先: {args.result_dir}/summary.csv, {args.fig_dir}/summary_*.png")


if __name__ == "__main__":
    main()
