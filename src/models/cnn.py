"""
========================================================================
アブレーション用 1D CNN (1D CNN for the Ablation Study)
========================================================================

仕様書4.1節で固定したアーキテクチャを実装する．条件B/C/Dで共通の
構造を使い，**入力チャンネル数と補助入力の有無だけ**が変わるように
設計している（4.0節の公平な比較の要請）．

■ 構造
    入力: (バッチ, 窓幅N, チャンネル数)  ← 条件により1 or 6 or 7
      → Conv1D ステム（チャンネル数を hidden に揃える）
      → 残差ブロック × (n_blocks - 1)．各ブロック後に長さを1/2へ縮約
      → Flatten または Global Average Pooling
      → [条件C・Dのみ] 時刻t+hの負荷ベクトル(6次元)を連結
      → 全結合層 2層
      → 出力: OT(t+h) の予測値（スカラー1つ）

■ 設計判断
・残差接続: 仕様書4.1節の選定理由（ResNetの研究経験を活かす）に対応する．
・Flattenを既定とする理由: EDAの最重要示唆は「直近のOTが支配的」であり，
  GAPは窓全体を平均するため時刻位置の情報が失われ，直近値を重く見る
  ことができない．一方で全長96をそのまま展開すると全結合層が肥大化する
  ため，ブロックごとにプーリングで長さを縮約してから展開する．
    - 比較用に --pooling gap も選べるようにしてある．

■ 動作確認
$ python src/models/cnn.py
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Conv1D 2層＋残差接続．チャンネル数は変えず，最後に長さを1/2へ縮約する．"""

    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.dropout(h)
        h = self.bn2(self.conv2(h))
        return self.pool(self.relu(x + h))


class AblationCNN(nn.Module):
    """
    条件B/C/D共通の1D CNN．

    ・n_channels: 窓入力のチャンネル数（B=1, C=7, D=6）
    ・n_aux: 時刻t+hの補助入力の次元数（B=0, C=6, D=6）
    ・条件間で差が出るのは，ステム畳み込みの入力側と全結合層の入力側だけ．
      層数・チャンネル幅・カーネルサイズはすべて共通．
    """

    def __init__(
        self,
        n_channels: int,
        n_aux: int,
        window: int,
        hidden: int = 32,
        n_blocks: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.1,
        pooling: str = "flatten",
    ):
        super().__init__()
        if pooling not in ("flatten", "gap"):
            raise ValueError(f"unknown pooling: {pooling}")
        self.pooling = pooling

        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, hidden, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden, kernel_size, dropout) for _ in range(n_blocks - 1)]
        )

        # 各ブロックで長さが半分になるため，展開後の次元数を事前に求めておく．
        pooled_len = window
        for _ in range(n_blocks - 1):
            pooled_len //= 2
        if pooled_len < 1:
            raise ValueError(f"window={window} is too short for n_blocks={n_blocks}")

        head_in = (hidden * pooled_len if pooling == "flatten" else hidden) + n_aux
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        """
        ・x_seq: (バッチ, 窓幅N, チャンネル数)
        ・x_aux: (バッチ, n_aux)．条件Bでは (バッチ, 0) が渡る
        ・戻り値: (バッチ,) の予測値（正規化スケール）
        """
        # Conv1dは (バッチ, チャンネル, 長さ) を期待するため軸を入れ替える．
        h = self.stem(x_seq.transpose(1, 2))
        for block in self.blocks:
            h = block(h)

        h = h.flatten(1) if self.pooling == "flatten" else h.mean(dim=2)
        if x_aux.shape[1] > 0:
            h = torch.cat([h, x_aux], dim=1)
        return self.head(h).squeeze(1)


def count_parameters(model: nn.Module) -> int:
    """学習対象パラメータ数．条件間の公平性を確認するために使う．"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    """動作確認用: 条件ごとの出力形状とパラメータ数を表示する．"""
    window, batch = 96, 4
    specs = {"B": (1, 0), "C": (7, 6), "D": (6, 6)}
    for pooling in ("flatten", "gap"):
        print(f"=== pooling={pooling} (window={window}) ===")
        for cond, (n_ch, n_aux) in specs.items():
            model = AblationCNN(n_ch, n_aux, window=window, pooling=pooling)
            out = model(torch.randn(batch, window, n_ch), torch.randn(batch, n_aux))
            print(
                f"  条件{cond}: channels={n_ch} aux={n_aux} "
                f"out={tuple(out.shape)} params={count_parameters(model):,}"
            )


if __name__ == "__main__":
    main()
