"""
Decoupled Classifier-Free Guidance (DCFG) for ReStyle-TTS

DCFG separates text and reference audio guidance to reduce dependency on reference style.

Formula:
    f̂ = f_{∅,t} + λ_t(f_{∅,t} - f_{∅,∅}) + λ_a(f_{a,t} - f_{∅,t})

Where:
    - f_{a,t}: Full conditional prediction (audio + text)
    - f_{∅,t}: Text-only prediction (no audio)
    - f_{∅,∅}: Unconditional prediction (no audio, no text)
    - λ_t: Text guidance strength (default: 2.0)
    - λ_a: Audio guidance strength (default: 0.5)

Reference: ReStyle-TTS (arXiv:2601.03632) Section 3.1
"""

from dataclasses import dataclass

import torch


@dataclass
class DCFGConfig:
    """DCFG (Decoupled Classifier-Free Guidance) の設定

    Attributes:
        lambda_t: テキストガイダンス強度 (デフォルト: 2.0)
        lambda_a: 参照音声ガイダンス強度 (デフォルト: 0.5)
        enabled: DCFGを有効にするか (デフォルト: True)
    """

    lambda_t: float = 2.0
    lambda_a: float = 0.5
    enabled: bool = True


def dcfg_combine(
    f_full: torch.Tensor,
    f_text: torch.Tensor,
    f_null: torch.Tensor,
    lambda_t: float = 2.0,
    lambda_a: float = 0.5,
) -> torch.Tensor:
    """DCFG式による予測の合成

    3つの予測を組み合わせて、テキストと参照音声のガイダンスを分離した予測を生成する。

    Args:
        f_full: フル条件予測 f_{a,t} (参照音声 + テキスト)
        f_text: テキストのみ予測 f_{∅,t} (参照音声なし)
        f_null: 無条件予測 f_{∅,∅} (参照音声なし、テキストなし)
        lambda_t: テキストガイダンス強度 (デフォルト: 2.0)
        lambda_a: 参照音声ガイダンス強度 (デフォルト: 0.5)

    Returns:
        DCFG式で合成された予測テンソル

    Note:
        - lambda_a=0.0: 参照音声の影響なし（テキストのみで生成）
        - lambda_a=1.0: 参照音声の影響を増加
        - 従来CFGと等価: lambda_t=cfg_strength, lambda_a=1+cfg_strength
          (ただしf_full=f_textの場合)
    """
    # DCFG式: f̂ = f_t + λ_t(f_t - f_∅) + λ_a(f_at - f_t)
    return f_text + lambda_t * (f_text - f_null) + lambda_a * (f_full - f_text)
