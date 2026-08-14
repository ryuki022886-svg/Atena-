"""
ETT オイル温度予測 PoC のソースコード一式．

■ パッケージの役割分担
・common: 全フェーズが共有する土台（データ構築・指標・描画スタイル）
・analysis: Phase 1．データを観察して示唆を得る
・train: Phase 2．アブレーション実験を実行して予測値を書き出す
・evaluation: Phase 2．書き出された予測値を集計して報告用の図表にする

■ 実行方法
リポジトリのルートからモジュールとして呼び出す．
$ python -m src.train.ablation --conditions B C D
"""
