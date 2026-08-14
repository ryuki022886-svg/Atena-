# ETT オイル温度予測 PoC

Athena Technologies社 インターン選考課題。ETT（Electricity Transformer Temperature）データセットを用いて、
変圧器のオイル温度（OT）を将来予測し、実際の運用・保全判断にどの程度価値を出せそうかを検証するPoC。

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU版を明示
```

## リポジトリ構成

```
data/                  ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv
src/
  eda.py               EDA（仕様書3節の6分析）
  trend.py             経年トレンドの可視化（同月の年度間比較）
  dataset.py           窓生成・分割・正規化（全実験の共通基盤）
  metrics.py           評価指標（MAE / RMSE）
  baseline.py          条件A: Persistenceベースライン
  models/cnn.py        1D CNN（条件B/C/D共通のアーキテクチャ）
  train.py             学習ループ
  evaluate.py          結果の集計と可視化
  operational.py       運用価値の評価（リードタイム・誤差の裾）
figures/               図
results/               数値結果（CSV）と予測値（npz）
```

## 実行方法

```bash
# Phase 1: EDA
python src/eda.py --datasets ETTh1 ETTh2
python src/trend.py

# Phase 2: ベースラインと学習
python src/baseline.py --split-mode informer
python src/baseline.py --split-mode fiscal
python src/train.py --conditions B C D --target-mode absolute --split-mode informer
python src/train.py --conditions B C D --target-mode delta    --split-mode informer
python src/train.py --conditions B C D --target-mode delta    --split-mode fiscal

# 集計・評価
python src/evaluate.py
python src/operational.py --pred-subdir preds_informer_delta --suffix _informer_delta
```

## 実験設計

### アブレーション条件（仕様書4.0節）

同一のモデル構造で**入力特徴量セットだけを変える**ことで、公平な比較を担保する。

| 条件 | 入力 | パラメータ数 | 検証内容 |
|---|---|---|---|
| A | なし（直前値をそのまま出力） | 学習なし | 比較の基準線 |
| B | OTの過去96時刻 | 70,081 | 自己相関だけでどこまで予測できるか |
| C | OT＋負荷6列の過去96時刻 ＋ 時刻t+hの負荷 | 71,425 | 負荷を足すと改善するか |
| D | 負荷6列の過去96時刻 ＋ 時刻t+hの負荷 | 71,265 | 負荷単体の限界 |

パラメータ数の差は2%以内。層数・チャンネル幅・カーネルサイズは全条件で共通。

### 予測対象の2つの定式化

| | 内容 |
|---|---|
| absolute | OT(t+h) を直接予測する。仕様書どおりの素直な実装 |
| delta | OT(t+h) − OT(t) を予測する。入力窓・補助入力からも予測実行時点tの値を引く |

### 分割の2つの方法

| | train | val | test |
|---|---|---|---|
| informer | 12ヶ月 | 4ヶ月 | 4ヶ月（冬季・2,880時刻） |
| fiscal | 年度1の10ヶ月 | 年度1の2ヶ月 | 年度2の12ヶ月（全季節・8,660時刻） |

正規化はいずれもtrain区間の統計のみを使用（リーク防止）。

## 主な結果

### Phase 1: EDA

- OTの自己相関は1h先で0.994と極めて強く支配的。168h後でも0.83〜0.87、720h後でも0.72〜0.78
- 負荷特徴量とOTの相関は最大でも0.5以下（ETTh1は最大0.22、ETTh2は最大0.49）
- 周期性は年次 >> 日内 > 曜日
- ETTh2のMUFLで1,025時刻（約42日）連続で完全同一値 → センサ固着の疑い
- Persistenceベースラインは短期（1h）で既に高精度（MAE 0.61〜0.89）

### 分布シフトの正体は「経年変化」

同じ月を年度間で比較すると（`figures/trend_year_over_year.png`）：

| | 年度1平均 | 年度2平均 | 差 |
|---|---|---|---|
| ETTh1 | 17.17℃ | 9.44℃ | **−7.71℃（12ヶ月すべてで低下）** |
| ETTh2 | 27.02℃ | 26.19℃ | −0.71℃（月により上下） |

ETTh1は季節性では説明できない系統的な低下がある。原因はデータからは特定できない。
ドリフトは油温だけでなく負荷にも現れ、**ETTh1は油温経由（−0.90σ）、ETTh2は負荷経由（最大−1.47σ）**と
入る経路が異なる。

### Phase 2: アブレーション（Persistence比、1.0未満で勝ち）

`figures/summary_ratio.png` を参照。

```
              1h    24h    96h   336h        1h    24h    96h   336h
           ┌─ informer / absolute ─┐      ┌── informer / delta ──┐
ETTh1 B    1.12   1.59   1.96   1.31       0.99   0.96   0.99   0.86
ETTh1 C    2.29   4.58   3.89   2.34       1.06   1.01   0.98   1.02
ETTh1 D   32.64   9.67   7.10   5.61       1.00   1.04   1.06   1.05
ETTh2 B    0.50   1.00   1.22   0.98       0.43   0.98   0.93   0.82
ETTh2 C    1.86   2.57   2.24   1.51       0.54   0.99   1.01   0.99
ETTh2 D   16.24   4.22   3.89   2.21       0.76   1.09   1.07   1.04
```

1. **素直な実装（absolute）はPersistenceに8ケース中6敗**。原因は分布シフトで、モデルが自信のない
   場面でtrain期間の水準（ETTh1で17.13℃）に予測を寄せるため。test期間の平均は4.85℃であり大きく外す
2. **予測対象をΔに変えると条件Bが8戦全勝**。モデル構造・特徴量は一切変えていない。
   最大の改善は ETTh1 条件C 24h の 6.989 → 1.543（誤差1/4.5）
3. **負荷特徴量は寄与しない**。条件C ÷ 条件B はΔ版で1.0前後（足しても引いても変わらない）。
   条件Dは8ケース中7ケースでPersistenceに負け、「変化しない」という仮定すら超えられない
4. **仕様書4.0節の仮説「長期になるほど負荷の寄与が大きくなる」は成立しなかった**。
   336hでむしろ最も悪化する（1.02 / 0.99）

年度分割（test 12ヶ月・全季節）でも同じ順位が再現され、季節に偏った評価ではないことを確認した。

### 運用価値：リードタイム

許容誤差を満たせる最長ホライズン（＝何時間先まで実用に耐えるか）。

| | 誤差1℃以内 | 誤差2℃以内 | 誤差3℃以内 |
|---|---|---|---|
| ETTh1 | 1時間先 | 24時間先 | 336時間（2週間）先 |
| ETTh2 | 1時間先 | 1時間先 | 24時間先 |

変圧器によって実用可能な予見期間が1桁異なる。ETTh2のほうがOTの変動が大きい（標準偏差12.2℃ vs 5.8℃）ため。

## 設計上の判断

- **アーキテクチャを1D CNNに固定**（仕様書4.1節）。テーマは「モデルの複雑さ」ではなく「特徴量セットの寄与」であるため
- **正規化層を置かない**。BatchNormはチャンネルごとに平均を引くため、油温の絶対水準の情報を落とす。
  実測でも条件B・1hのtest MAEが 0.57（BNあり）→ 0.43（BNなし）と改善した
- **GAPではなくFlatten**。EDAの最重要示唆は「直近のOTが支配的」であり、GAPは窓全体を平均するため
  時刻位置の情報が失われる。ただし全長96を展開すると全結合層が肥大化するため、
  ブロックごとにプーリングで長さを1/2に縮約してから展開する
- **early stoppingはval MAEで判定**。分布シフトがあるため固定エポック数は危険

## 現時点での限界

- **乱数シードは1つ（42）のみ**。条件間の差が偶然でない保証は取れていない。
  ただし条件の順位は8ケースすべてで一貫している
- **絶対閾値による超過検知は評価できていない**。12/4/4分割ではtest区間が冬季にあたり、
  train区間の90パーセンタイル（ETTh1: 31.6℃）を超える時刻がtest区間に1つも存在しないため
- **モデル構造の探索は行っていない**。層数・チャンネル幅・窓幅Nは固定
- **ETTm1・ETTm2（15分粒度）はEDAのみ**。アブレーション実験はETTh1・ETTh2で実施（仕様書3.7節の判断）
