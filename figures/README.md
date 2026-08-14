# 図の索引

すべての図はスクリプトの実行で再生成される。手作業で編集したものはない。
ディレクトリの番号は読む順序（＝報告スライドの章立て）に対応する。

## 01_eda/ — データの観察

生成: `python -m src.analysis.eda --datasets ETTh1 ETTh2 ETTm1 ETTm2`
データセットごとにサブディレクトリを作る。ETTm1・ETTm2はEDAのみで、実験には使っていない。

| ファイル | 仕様書 | 内容 |
|---|---|---|
| `<データセット>/autocorrelation.png` | 3.1 | OTの自己相関（ラグ30日分）。1h先で0.994と自己相関が支配的 |
| `<データセット>/load_lag_correlation.png` | 3.2 | 負荷6列とOTのラグ相関（±24h）。負の遅れが強ければ負荷がOTに先行する |
| `<データセット>/periodicity.png` | 3.3 | 時刻別・曜日別・月別のOT平均。振幅の比較で年次 >> 日内 > 曜日と分かる |
| `<データセット>/distribution_shift.png` | 3.4 | 月次OT平均と12/4/4分割の位置。train/val/testで水準が違うことを示す |
| `<データセット>/persistence_baseline.png` | 3.6 | 全期間でのPersistence誤差。ホライズンごとの基準線 |

※ 仕様書3.5（データ品質）は表のみで図はない。コンソール出力を参照。

## 02_trend/ — 分布シフトの正体

生成: `python -m src.analysis.trend`

| ファイル | 内容 |
|---|---|
| `timeseries.png` | OTの全期間推移。12/4/4分割の帯と年度境界を重ねてある |
| `year_over_year.png` | 同じ月を年度1と年度2で比較。分布シフトが季節性ではなく経年変化であることの根拠 |

## 03_ablation/ — アブレーションの本体

生成: `python -m src.train.baseline`（persistence_baseline_*）と `python -m src.evaluation.evaluate`（それ以外）

| ファイル | 内容 |
|---|---|
| `persistence_baseline_informer.png` | 条件Aの誤差（12/4/4分割のtest区間）。CNNと同一サンプル集合で算出 |
| `persistence_baseline_fiscal.png` | 同上（年度分割のtest区間） |
| `mae.png` | ホライズン別のtest MAE。行＝定式化（分割×予測対象）、列＝データセット |
| `ratio.png` | **Persistence比のヒートマップ。1.0未満で条件Aに勝ち。結論を1枚で示す図** |
| `timeseries.png` | 予測と実測の重ね書き（ETTh1・24h・informer/delta・test先頭21日）。外し方の質を見る |

## 04_operational/ — 運用価値への翻訳

生成: `python -m src.evaluation.operational --pred-subdir preds_<分割>_<予測対象>`
接尾辞は予測ディレクトリ名から自動で決まる。

| ファイル | 内容 |
|---|---|
| `lead_time_<設定>.png` | ホライズン別MAEと許容誤差（1/2/3℃）の線。交点が実用可能なリードタイム |
| `spike_detection_<設定>.png` | 急変検知のrecall。条件Aは構造上0になるため、予測モデルを入れる意味が読める |

## 図のスタイルについて

条件の色と凡例は [src/common/plotting.py](../src/common/plotting.py) で固定している。
どの図でも 条件A＝黒破線 / B＝青 / C＝橙 / D＝緑 で一致するので、スライドに複数の図を
並べても同じ条件を追える。図中の文字は日本語フォントへの依存を避けるため英字にしている。
