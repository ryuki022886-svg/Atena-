"""
========================================================================
図の共通スタイル (Shared Plotting Style)
========================================================================

すべての図が同じ見た目になるよう，スタイルと保存処理を1箇所に集約する．

■ ここに集約する理由
・報告スライドには複数のスクリプトが出した図を並べる．条件Bが図ごとに
  違う色で描かれると，読み手が同じ条件を追えなくなる．色と凡例の文言を
  本モジュールで固定し，どの図でも一致させる．
・ホライズン軸（対数目盛＋実数表記）のような定型処理が各スクリプトに
  コピーされていたため，関数として1つにまとめる．

■ 図の配置
figures/ 配下をフェーズ順のディレクトリに分け，スライドの章立てと
対応させる．どの図がどのスクリプト由来かがパスから分かる．
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 画面のない環境で実行するため，描画先をファイルに固定する
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# figures/ 配下のフェーズ別サブディレクトリ．番号は読む順序を表す．
FIG_EDA = "01_eda"
FIG_TREND = "02_trend"
FIG_ABLATION = "03_ablation"
FIG_OPERATIONAL = "04_operational"

DPI = 120

# アブレーション条件の凡例．図中の文字は日本語フォントに依存しないよう英字で統一する．
CONDITION_LABEL = {
    "A": "A: Persistence",
    "B": "B: OT lag only",
    "C": "C: OT lag + load",
    "D": "D: load only",
}

# 条件ごとの線のスタイル．描画順や欠けた条件に左右されないよう明示的に固定する．
# 条件Aは基準線なので，学習ありの条件と区別できるよう黒の破線にする．
CONDITION_STYLE = {
    "A": {"color": "black", "linestyle": "--"},
    "B": {"color": "tab:blue"},
    "C": {"color": "tab:orange"},
    "D": {"color": "tab:green"},
}

# 分割・定式化を色で示すときの割り当て（複数設定を1枚に並べる図で使う）．
SPLIT_COLOR = {"informer": "tab:purple", "fiscal": "tab:brown"}

plt.rcParams.update({
    "figure.autolayout": False,   # tight_layout は save() 側で明示的に呼ぶ
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.grid": False,           # グリッドは grid() ヘルパーで明示的に付ける
})


def condition_kwargs(condition: str, **overrides) -> dict:
    """
    条件に対応する plot() のキーワード引数（色・線種・凡例）を返す．

    ・未知の条件が来ても落ちないよう，スタイル指定がなければ色は既定に任せる．
    """
    kwargs = {"label": CONDITION_LABEL.get(condition, f"cond {condition}")}
    kwargs.update(CONDITION_STYLE.get(condition, {}))
    kwargs.update(overrides)
    return kwargs


def horizon_axis(ax, horizons) -> None:
    """
    横軸をホライズン（時間）にする．1〜336hと幅が広いため対数目盛にし，
    目盛のラベルは指数表記ではなく実数（1, 24, 96, 336）で表示する．
    """
    ax.set_xscale("log")
    ax.set_xticks(sorted(set(int(h) for h in horizons)))
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel("horizon (hours)")


def grid(ax) -> None:
    """図の主張を邪魔しない濃さのグリッドを引く．"""
    ax.grid(alpha=0.3)


def save(fig, path: Path) -> Path:
    """
    図を保存して閉じる．保存先ディレクトリが無ければ作る．

    ・戻り値: 保存したパス（呼び出し側でそのままログに出せるようにする）
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path
