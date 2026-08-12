# ETT オイル温度予測 PoC

Athena Technologies社インターン選考課題。ETT（Electricity Transformer Temperature）データセットを用いて、
変圧器のオイル温度（OT）を将来予測するPoC。

## 進捗状況

- [x] Phase 1: EDA
- [ ] Phase 2: モデル構築
- [ ] Phase 3: 報告スライド

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## リポジトリ構成

```
data/       ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv
src/
  eda.py    EDA（自己相関・負荷特徴量相関・周期性・分布シフト・データ品質・Persistenceベースライン）
figures/    EDAの出力図（データセットごとにサブディレクトリ）
```

## EDAの実行

```bash
python src/eda.py --datasets ETTh1 ETTh2
```

## EDAの主な結果（ETTh1 / ETTh2）

- OTの自己相関は1h先で0.994と極めて強く支配的。168h(1週間)後でも0.83〜0.87、720h(1ヶ月)後でも
  0.72〜0.78と長いレンジまで相関が残る一方、負荷特徴量とOTの相関は最大でも0.5以下。
- 有効電力（UFL系）より無効電力（ULL系）の方がOTとの相関が高い（ETTh2: MULL=0.49 vs MUFL=0.19）。
- 周期性は年次 >> 日内 > 曜日（ETTh2は日内振幅9.37、14-15時にピーク）。
- ETTh1はtrain/testでOT分布が大きく異なる（train平均17.13 → test平均4.85）。
- ETTh2のMUFLで1,025時刻（約42日間）連続で完全同一値 → センサ固着の疑い。
- Persistenceベースライン（直近値予測）は短期ホライズン（1h）で既に高精度（MAE 0.61〜0.89）。
  ホライズンが長くなるほど誤差が拡大するため、深層学習モデルの価値は長期予測でこそ示せる可能性が高い。

詳細は `figures/` の各図を参照。
