"""
========================================================================
運用価値の評価 (Operational Value Evaluation)
========================================================================

課題資料の一次質問「オイル温度を将来予測することで，実際の運用・保全判断に
どの程度価値を出せそうか」に答えるための評価を行う．MAE/RMSEは精度の指標
ではあるが，保全判断に使えるかどうかを直接は語らないため，以下の3つの
運用寄りの軸に翻訳する．

■ 評価軸
1. リードタイム: 許容誤差を仮定したとき，何時間先まで実用に耐えるか
2. 誤差の裾: 平均ではなく「最悪どれだけ外すか」．保全判断では外れ方が効く
3. 急変検知: 「h時間後にΔ℃以上上昇する」事象を事前に検知できるか

■ 閾値に関する前提（課題資料に明記がないため置く仮定）
・資料には具体的な閾値水準の記載がないため，絶対閾値ではなく上記3軸で評価する．
・絶対閾値方式が使えない理由は本スクリプトが実測で示す．12/4/4分割ではtest区間が
  冬季にあたり，train区間の90パーセンタイル(ETTh1: 31.6℃)を超える時刻がtest区間に
  1つも存在しないため，超過検知の性能を測ることが原理的にできない．
    - これは仕様書3.4節の分布シフトの直接的な帰結であり，限界として報告する．
・急変検知の閾値Δは train区間の上昇幅分布の分位点から決める（testを見ないためリークなし）．

■ 出力（<tag> は予測ディレクトリ名から自動で決まる．例: informer_delta）
・results/operational_*_<tag>.csv
・figures/04_operational/lead_time_<tag>.png
・figures/04_operational/spike_detection_<tag>.png

■ 実行方法
$ python -m src.evaluation.operational --pred-subdir preds_informer_delta
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import plotting
from src.common.dataset import TARGET, load_dataframe, split_bounds, steps_per_hour


def load_predictions(pred_dir: Path) -> pd.DataFrame:
    """
    results/preds/ のnpzを読み込み，1行=1サンプルの長い表にまとめる．

    ・条件A（Persistence）は各npzに同梱されている persistence 列から復元する．
    """
    rows = []
    for path in sorted(pred_dir.glob("*.npz")):
        dataset, condition, horizon_tag, seed_tag = path.stem.split("_")
        data = np.load(path)
        common = {
            "dataset": dataset,
            "horizon": int(horizon_tag[1:]),
            "seed": int(seed_tag[1:]),
        }
        rows.append(pd.DataFrame({
            **common,
            "condition": condition,
            "true": data["true"],
            "pred": data["pred"],
            "origin": data["persistence"],
        }))
        # 条件Aは学習を伴わないが，同じサンプル集合の上で比較できるよう同じ表に載せる．
        rows.append(pd.DataFrame({
            **common,
            "condition": "A",
            "true": data["true"],
            "pred": data["persistence"],
            "origin": data["persistence"],
        }))

    table = pd.concat(rows, ignore_index=True)
    return table.drop_duplicates(subset=["dataset", "horizon", "seed", "condition", "true", "pred", "origin"])


def error_percentiles(table: pd.DataFrame) -> pd.DataFrame:
    """
    条件・ホライズン別に誤差の裾を集計する．

    ・MAEは平均なので「たまに大きく外す」性質が見えない．保全判断では
      その外れ方こそが問題になるため，絶対誤差の分位点を併記する．
    """
    table = table.assign(abs_error=(table["true"] - table["pred"]).abs())
    grouped = table.groupby(["dataset", "condition", "horizon"])["abs_error"]
    result = grouped.agg(
        mae="mean",
        p50=lambda s: s.quantile(0.50),
        p90=lambda s: s.quantile(0.90),
        p95=lambda s: s.quantile(0.95),
        p99=lambda s: s.quantile(0.99),
        worst="max",
    ).reset_index()
    return result.sort_values(["dataset", "condition", "horizon"])


def lead_time(errors: pd.DataFrame, tolerances: list) -> pd.DataFrame:
    """
    許容誤差ごとに「その精度を保てる最長ホライズン」を求める．

    ・保全計画は「何時間前に分かるか」で決まるため，精度を時間に翻訳する．
    ・どのホライズンでも許容誤差を満たせない場合は0hと表記する．
    """
    rows = []
    for (dataset, condition), group in errors.groupby(["dataset", "condition"]):
        group = group.sort_values("horizon")
        for tol in tolerances:
            ok = group[group["mae"] <= tol]["horizon"]
            rows.append({
                "dataset": dataset,
                "condition": condition,
                "tolerance": tol,
                "max_horizon": int(ok.max()) if len(ok) else 0,
            })
    return pd.DataFrame(rows)


def spike_thresholds(datasets: list, horizons: list, quantile: float, data_dir: Path) -> dict:
    """
    急変とみなす上昇幅Δを，train区間の上昇幅分布の分位点から決める．

    ・train区間のみから決めるためtestへのリークはない．
    ・戻り値: {(データセット名, ホライズン): Δ}
    """
    thresholds = {}
    for name in datasets:
        df = load_dataframe(name, data_dir)
        sph = steps_per_hour(df)
        lo, hi = split_bounds(len(df), sph)["train"]
        ot = df[TARGET].to_numpy()[lo:hi]
        for h in horizons:
            step = h * sph
            thresholds[(name, h)] = float(np.quantile(ot[step:] - ot[:-step], quantile))
    return thresholds


def spike_detection(table: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    「h時間後にΔ℃以上上昇する」事象を事前に検知できるかを評価する．

    ・実測の上昇幅 = OT(t+h) - OT(t)，予測の上昇幅 = 予測値 - OT(t)
    ・条件A（Persistence）は予測上昇幅が常に0になるため，構造的に検知できない．
      この対比が「予測モデルを入れる意味」を最も端的に示す．
    ・recall: 実際に起きた急変のうち検知できた割合（見逃しの少なさ）
    ・precision: 検知と言った中で実際に急変だった割合（誤報の少なさ）
    """
    rows = []
    for (dataset, condition, horizon), group in table.groupby(["dataset", "condition", "horizon"]):
        delta = thresholds[(dataset, horizon)]
        actual = (group["true"] - group["origin"]).to_numpy() >= delta
        detected = (group["pred"] - group["origin"]).to_numpy() >= delta

        tp = int(np.sum(actual & detected))
        fp = int(np.sum(~actual & detected))
        fn = int(np.sum(actual & ~detected))
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall and not np.isnan(precision) and not np.isnan(recall)
              else 0.0)

        rows.append({
            "dataset": dataset,
            "condition": condition,
            "horizon": horizon,
            "delta": delta,
            "n_events": int(actual.sum()),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return pd.DataFrame(rows).sort_values(["dataset", "horizon", "condition"])


def check_absolute_threshold(datasets: list, quantiles: list, data_dir: Path) -> pd.DataFrame:
    """
    絶対閾値による超過検知が本分割で評価可能かを確認する．

    ・train区間の分位点を閾値とし，test区間に超過時刻がいくつあるかを数える．
    ・0件であれば絶対閾値方式は評価不能であり，その事実自体を報告する．
    """
    rows = []
    for name in datasets:
        df = load_dataframe(name, data_dir)
        sph = steps_per_hour(df)
        bounds = split_bounds(len(df), sph)
        train_ot = df[TARGET].iloc[slice(*bounds["train"])]
        test_ot = df[TARGET].iloc[slice(*bounds["test"])]
        for q in quantiles:
            threshold = float(np.percentile(train_ot, q))
            rows.append({
                "dataset": name,
                "train_quantile": q,
                "threshold": threshold,
                "test_max": float(test_ot.max()),
                "test_exceedances": int((test_ot > threshold).sum()),
                "test_n": len(test_ot),
            })
    return pd.DataFrame(rows)


def plot_lead_time(errors: pd.DataFrame, tolerances: list, fig_path: Path) -> Path:
    """ホライズンに対するMAEの伸びを，許容誤差の線とあわせて描く．"""
    datasets = sorted(errors["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 4), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        subset = errors[errors["dataset"] == dataset]
        for condition, group in subset.groupby("condition"):
            group = group.sort_values("horizon")
            ax.plot(group["horizon"], group["mae"], marker="o",
                    **plotting.condition_kwargs(condition))
        for tol in tolerances:
            ax.axhline(tol, color="gray", linestyle=":", linewidth=0.8)
            ax.text(1, tol, f" tolerance {tol}degC", va="bottom", fontsize=8, color="gray")
        plotting.horizon_axis(ax, subset["horizon"].unique())
        ax.set_ylabel("test MAE (degC)")
        ax.set_title(f"{dataset}: usable lead time")
        plotting.grid(ax)
        ax.legend()
    return plotting.save(fig, fig_path)


def plot_spike_recall(spikes: pd.DataFrame, fig_path: Path) -> Path:
    """急変検知のrecallを条件別・ホライズン別に描く．"""
    datasets = sorted(spikes["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 4), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        subset = spikes[spikes["dataset"] == dataset]
        for condition, group in subset.groupby("condition"):
            group = group.sort_values("horizon")
            ax.plot(group["horizon"], group["recall"], marker="o",
                    **plotting.condition_kwargs(condition))
        plotting.horizon_axis(ax, subset["horizon"].unique())
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("recall of temperature spikes")
        ax.set_title(f"{dataset}: spike detection")
        plotting.grid(ax)
        ax.legend()
    return plotting.save(fig, fig_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="予測精度を運用価値の指標へ翻訳する")
    parser.add_argument("--tolerances", type=float, nargs="+", default=[1.0, 2.0, 3.0],
                        help="保全判断で許容できる誤差の仮定（摂氏）")
    parser.add_argument("--spike-quantile", type=float, default=0.95,
                        help="急変とみなす上昇幅の分位点（train区間から算出）")
    parser.add_argument("--pred-subdir", default="preds_informer_delta",
                        help="result-dir配下の予測ディレクトリ（例: preds_informer_delta）")
    parser.add_argument("--suffix", default=None,
                        help="出力ファイル名に付ける接尾辞（既定は予測ディレクトリ名から決める）")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--result-dir", type=Path, default=Path("results"))
    parser.add_argument("--fig-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    pred_dir = args.result_dir / args.pred_subdir
    if not any(pred_dir.glob("*.npz")):
        raise SystemExit(
            f"予測が見つかりません: {pred_dir}"
            "（先に python -m src.train.ablation を実行してください）"
        )

    # 入力の予測ディレクトリと出力名を必ず対応させ，設定違いの結果が混ざらないようにする．
    suffix = args.suffix if args.suffix is not None else "_" + args.pred_subdir.removeprefix("preds_")

    table = load_predictions(pred_dir)
    datasets = sorted(table["dataset"].unique())
    horizons = sorted(table["horizon"].unique())

    # 1. 絶対閾値方式が使えるかの確認
    absolute = check_absolute_threshold(datasets, [90, 95, 99], args.data_dir)
    print("=== 1. 絶対閾値による超過検知が評価可能か ===")
    for _, r in absolute.iterrows():
        print(f"  {r.dataset} train{int(r.train_quantile)}%点={r.threshold:5.1f}℃  "
              f"test最高={r.test_max:5.1f}℃  超過時刻={int(r.test_exceedances)}/{int(r.test_n)}")
    if absolute["test_exceedances"].sum() == 0:
        print("  → test区間に超過事象が1つも存在せず，絶対閾値方式では評価不能．")
        print("    仕様書3.4節の分布シフト（test区間が冬季）の帰結．限界として報告する．")

    # 2. 誤差の裾とリードタイム
    errors = error_percentiles(table)
    leads = lead_time(errors, args.tolerances)
    print("\n=== 2. 誤差の裾（test区間，摂氏） ===")
    print(f"  {'dataset':7s} {'cond':>4s} {'horizon':>8s} {'MAE':>6s} {'p90':>6s} {'p95':>6s} {'p99':>6s} {'最悪':>6s}")
    for _, r in errors.iterrows():
        print(f"  {r.dataset:7s} {r.condition:>4s} {r.horizon:7d}h "
              f"{r.mae:6.2f} {r.p90:6.2f} {r.p95:6.2f} {r.p99:6.2f} {r.worst:6.2f}")

    print("\n=== 3. リードタイム（許容誤差を満たせる最長ホライズン） ===")
    pivot = leads.pivot_table(index=["dataset", "condition"], columns="tolerance", values="max_horizon")
    print(pivot.to_string())

    # 3. 急変検知
    thresholds = spike_thresholds(datasets, horizons, args.spike_quantile, args.data_dir)
    spikes = spike_detection(table, thresholds)
    print(f"\n=== 4. 急変検知（train上昇幅の{args.spike_quantile:.0%}点をΔとする） ===")
    print(f"  {'dataset':7s} {'horizon':>8s} {'Δ':>6s} {'cond':>4s} {'事象数':>6s} "
          f"{'recall':>7s} {'precis':>7s} {'F1':>6s}")
    for _, r in spikes.iterrows():
        print(f"  {r.dataset:7s} {r.horizon:7d}h {r.delta:6.2f} {r.condition:>4s} "
              f"{int(r.n_events):6d} {r.recall:7.3f} {r.precision:7.3f} {r.f1:6.3f}")

    absolute.to_csv(args.result_dir / f"operational_absolute_threshold{suffix}.csv", index=False)
    errors.to_csv(args.result_dir / f"operational_error_percentiles{suffix}.csv", index=False)
    leads.to_csv(args.result_dir / f"operational_lead_time{suffix}.csv", index=False)
    spikes.to_csv(args.result_dir / f"operational_spike_detection{suffix}.csv", index=False)

    outdir = args.fig_dir / plotting.FIG_OPERATIONAL
    plot_lead_time(errors, args.tolerances, outdir / f"lead_time{suffix}.png")
    plot_spike_recall(spikes, outdir / f"spike_detection{suffix}.png")
    print(f"\n保存先: {args.result_dir}/operational_*{suffix}.csv, {outdir}/")


if __name__ == "__main__":
    main()
