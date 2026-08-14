"""
========================================================================
アブレーション実験の学習・評価 (Training for the Ablation Study)
========================================================================

仕様書4.3節 Step 2 に対応する．条件B・C・Dの1D CNNを，複数ホライズン・
複数データセットについて学習し，条件A（Persistence）と同一のtest区間で
評価した結果を results/ablation.csv に書き出す．

■ 学習設定
・損失: MSE（正規化スケール上で計算）
・最適化: Adam
・early stopping: val区間のMAE（摂氏スケール）が改善しなくなったら打ち切る
    - 分布シフトがあるため固定エポック数は危険であり，val基準で止める
・評価: 最良エポックの重みに戻してからtest区間を推論し，摂氏スケールでMAE/RMSEを算出

■ 公平性のための約束
・条件間で変えるのは入力特徴量セットのみ．学習率・バッチサイズ・エポック上限・
  early stoppingの条件・乱数シードはすべて共通とする．
・同一シードで条件B/C/Dを回すため，条件間の差がシード差に埋もれない．

■ 実行方法
$ python src/train.py --datasets ETTh1 ETTh2 --conditions B C D
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import Bundle, SplitData, build_bundle
from metrics import score
from models.cnn import AblationCNN, count_parameters


def set_seed(seed: int) -> None:
    """乱数シードを固定し，条件間の比較が実行ごとにブレないようにする．"""
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensors(split: SplitData) -> tuple:
    """SplitDataをPyTorchのテンソルへ変換する．"""
    return (
        torch.from_numpy(split.x_seq),
        torch.from_numpy(split.x_aux),
        torch.from_numpy(split.y),
    )


@torch.no_grad()
def predict(model: nn.Module, split: SplitData, bundle: Bundle, batch_size: int) -> np.ndarray:
    """
    分割全体を推論し，摂氏スケールに戻した予測値を返す．

    ・戻り値: (サンプル数,) の予測値（摂氏）
    """
    model.eval()
    x_seq, x_aux, _ = to_tensors(split)
    outputs = []
    for i in range(0, len(x_seq), batch_size):
        outputs.append(model(x_seq[i : i + batch_size], x_aux[i : i + batch_size]))
    return bundle.inverse_target(torch.cat(outputs).numpy(), split.origin_ot)


def train_one(bundle: Bundle, args: argparse.Namespace, seed: int) -> dict:
    """
    1つの (データセット, 条件, ホライズン, シード) を学習し，評価結果を返す．

    ・戻り値: 指標・学習エポック数・パラメータ数などをまとめた辞書
    ・予測値は呼び出し側で保存できるよう "test_pred" キーに同梱する
    """
    set_seed(seed)
    model = AblationCNN(
        n_channels=bundle.n_channels,
        n_aux=bundle.n_aux,
        window=bundle.window,
        hidden=args.hidden,
        n_blocks=args.blocks,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        pooling=args.pooling,
        norm=args.norm,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(*to_tensors(bundle.train)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x_seq, x_aux, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x_seq, x_aux), y)
            loss.backward()
            optimizer.step()

        # val区間のMAE（摂氏）で最良エポックを選ぶ．
        val_mae = score(bundle.val.y_raw, predict(model, bundle.val, bundle, args.batch_size))["mae"]
        if val_mae < best_val - args.min_delta:
            best_val = val_mae
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            break

    model.load_state_dict(best_state)
    test_pred = predict(model, bundle.test, bundle, args.batch_size)
    test_scores = score(bundle.test.y_raw, test_pred)

    return {
        "dataset": bundle.dataset,
        "condition": bundle.condition,
        "horizon": bundle.horizon,
        "window": bundle.window,
        "split_mode": bundle.split_mode,
        "target_mode": bundle.target_mode,
        "seed": seed,
        "val_mae": best_val,
        "test_mae": test_scores["mae"],
        "test_rmse": test_scores["rmse"],
        "best_epoch": best_epoch,
        "params": count_parameters(model),
        "seconds": time.time() - started,
        "test_pred": test_pred,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="1D CNNによるアブレーション実験の学習")
    parser.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2"])
    parser.add_argument("--conditions", nargs="+", default=["B", "C", "D"])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 24, 96, 336])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--window", type=int, default=96, help="入力窓幅N（時刻数）")
    parser.add_argument("--target-mode", default="absolute", choices=["absolute", "delta"],
                        help="absolute: OT(t+h)を予測 / delta: OT(t+h)-OT(t)を予測")
    parser.add_argument("--split-mode", default="informer", choices=["informer", "fiscal"],
                        help="informer: 12/4/4ヶ月分割 / fiscal: 年度1で学習し年度2でテスト")

    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", default="flatten", choices=["flatten", "gap"])
    parser.add_argument("--norm", default="none", choices=["none", "batch"],
                        help="正規化層．BatchNormは絶対水準の情報を落とすため既定はnone")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100, help="エポック数の上限")
    parser.add_argument("--patience", type=int, default=10, help="early stoppingの猶予エポック数")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="改善とみなす最小幅")

    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--result-dir", type=Path, default=Path("results"))
    parser.add_argument("--tag", default=None,
                        help="出力名の識別子（既定は <split_mode>_<target_mode>）")
    parser.add_argument("--out", default=None, help="results-dir配下の出力CSV名")
    args = parser.parse_args()

    # 設定ごとに出力を分け，複数の定式化を並べて比較できるようにする．
    tag = args.tag or f"{args.split_mode}_{args.target_mode}"
    if args.out is None:
        args.out = f"ablation_{tag}.csv"
    pred_dir = args.result_dir / f"preds_{tag}"
    pred_dir.mkdir(parents=True, exist_ok=True)

    total = len(args.datasets) * len(args.conditions) * len(args.horizons) * len(args.seeds)
    print(f"学習対象: {total}モデル (split={args.split_mode}, target={args.target_mode}, "
          f"pooling={args.pooling}, window={args.window})")
    header = f"{'dataset':7s} {'cond':>4s} {'horizon':>8s} {'seed':>5s} " \
             f"{'val MAE':>8s} {'test MAE':>9s} {'test RMSE':>10s} {'epoch':>6s} {'sec':>6s}"
    print(header)

    rows = []
    for name in args.datasets:
        for cond in args.conditions:
            for horizon in args.horizons:
                bundle = build_bundle(name, cond, horizon, args.window, args.data_dir,
                                      args.target_mode, args.split_mode)
                for seed in args.seeds:
                    result = train_one(bundle, args, seed)
                    test_pred = result.pop("test_pred")
                    np.savez_compressed(
                        pred_dir / f"{name}_{cond}_h{horizon}_s{seed}.npz",
                        pred=test_pred,
                        true=bundle.test.y_raw,
                        persistence=bundle.test.origin_ot,
                        target_index=bundle.test.target_index,
                    )
                    rows.append(result)
                    print(
                        f"{name:7s} {cond:>4s} {horizon:7d}h {seed:5d} "
                        f"{result['val_mae']:8.3f} {result['test_mae']:9.3f} "
                        f"{result['test_rmse']:10.3f} {result['best_epoch']:6d} "
                        f"{result['seconds']:6.1f}"
                    )

    csv_path = args.result_dir / args.out
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n保存先: {csv_path}, {pred_dir}/")


if __name__ == "__main__":
    main()
