"""
Phase 2: 実験の実行 (Experiment runners).

・cnn: 条件B/C/Dで共通の1D CNN
・baseline: 条件A（Persistence）．学習は伴わないが，CNNとまったく同じ
  サンプル集合の上で基準線を作るため実験の実行側に置く
・ablation: 条件B/C/Dの学習ループ

いずれも results/ に数値と予測値を書き出すところまでを担当し，
集計と図表化は evaluation パッケージが受け持つ．
"""
