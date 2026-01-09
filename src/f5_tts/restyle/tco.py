"""
TCO (Timbre Consistency Optimization) for ReStyle-TTS

DCFGで参照音声への依存を減らした際の音色劣化を補償する。
アドバンテージ重み付きFlow Matching損失を使用。

Reference: ReStyle-TTS (arXiv:2601.03632) Section 3.4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable

import torch
import torch.nn as nn


@dataclass
class TCOConfig:
    """TCOの設定

    Attributes:
        lambda_reward: 報酬強度 (デフォルト: 0.2)
        beta: アドバンテージ感度 (デフォルト: 5.0)
        mu: EMAモメンタム (デフォルト: 0.9)
        min_weight: 最小重み (デフォルト: 0.1)
        max_weight: 最大重み (デフォルト: 2.0)
        enabled: TCOを有効にするか (デフォルト: True)
    """

    lambda_reward: float = 0.2
    beta: float = 5.0
    mu: float = 0.9
    min_weight: float = 0.1
    max_weight: float = 2.0
    enabled: bool = True


class TCOWeightComputer(nn.Module):
    """TCO重み計算モジュール

    アドバンテージベースの重みを計算する。

    重み計算式:
        w_t = 1 + λ * tanh(β * A_t)
        A_t = r_t - b_t  (アドバンテージ = 報酬 - ベースライン)

    Usage:
        >>> computer = TCOWeightComputer()
        >>> rewards = torch.tensor([0.8, 0.6, 0.9])
        >>> weights = computer(rewards)
    """

    def __init__(self, config: Optional[TCOConfig] = None):
        """
        Args:
            config: TCO設定
        """
        super().__init__()
        self.config = config or TCOConfig()

        # EMAベースラインをバッファとして登録
        self.register_buffer("baseline", torch.tensor(0.5))
        self.register_buffer("num_updates", torch.tensor(0))

    def update_baseline(self, reward: torch.Tensor) -> None:
        """EMAベースラインを更新

        b_t = μ * b_{t-1} + (1 - μ) * r_t

        Args:
            reward: 現在の報酬（バッチ平均）
        """
        reward_mean = reward.mean().detach()

        if self.num_updates == 0:
            # 初回は報酬をそのままベースラインに設定
            self.baseline = reward_mean
        else:
            # EMA更新
            self.baseline = self.config.mu * self.baseline + (1 - self.config.mu) * reward_mean

        self.num_updates += 1

    def compute_advantage(self, reward: torch.Tensor) -> torch.Tensor:
        """アドバンテージを計算

        A_t = r_t - b_t

        Args:
            reward: 報酬 [B]

        Returns:
            アドバンテージ [B]
        """
        return reward - self.baseline

    def compute_weight(self, advantage: torch.Tensor) -> torch.Tensor:
        """重みを計算

        w_t = 1 + λ * tanh(β * A_t)

        Args:
            advantage: アドバンテージ [B]

        Returns:
            重み [B]
        """
        weight = 1.0 + self.config.lambda_reward * torch.tanh(self.config.beta * advantage)

        # 重みをクリップ
        weight = weight.clamp(min=self.config.min_weight, max=self.config.max_weight)

        return weight

    def forward(self, reward: torch.Tensor, update_baseline: bool = True) -> torch.Tensor:
        """報酬から重みを計算

        Args:
            reward: 話者類似度報酬 [B]
            update_baseline: ベースラインを更新するか

        Returns:
            損失の重み [B]
        """
        if not self.config.enabled:
            return torch.ones_like(reward)

        # アドバンテージ計算
        advantage = self.compute_advantage(reward)

        # 重み計算
        weight = self.compute_weight(advantage)

        # ベースライン更新
        if update_baseline and self.training:
            self.update_baseline(reward)

        return weight


class TCOLoss(nn.Module):
    """TCO重み付き損失

    L_total = w_t * L_FM

    Usage:
        >>> tco_loss = TCOLoss(speaker_encoder)
        >>> loss = tco_loss(base_loss, generated_audio, reference_audio)
    """

    def __init__(
        self,
        speaker_encoder: Optional[nn.Module] = None,
        config: Optional[TCOConfig] = None,
    ):
        """
        Args:
            speaker_encoder: 話者エンコーダー（Noneの場合は遅延初期化）
            config: TCO設定
        """
        super().__init__()
        self.config = config or TCOConfig()
        self.weight_computer = TCOWeightComputer(config)
        self._speaker_encoder = speaker_encoder

    @property
    def speaker_encoder(self) -> nn.Module:
        """話者エンコーダーを取得（遅延初期化）"""
        if self._speaker_encoder is None:
            from f5_tts.restyle.speaker_encoder import SpeakerEncoder
            self._speaker_encoder = SpeakerEncoder()
        return self._speaker_encoder

    def compute_reward(
        self,
        generated_audio: torch.Tensor,
        reference_audio: torch.Tensor,
        sample_rate: int = 24000,
    ) -> torch.Tensor:
        """話者類似度報酬を計算

        Args:
            generated_audio: 生成音声 [B, T]
            reference_audio: 参照音声 [B, T]
            sample_rate: サンプリングレート

        Returns:
            話者類似度 [B]
        """
        # 話者埋め込みを計算
        gen_emb = self.speaker_encoder(generated_audio, sample_rate)
        ref_emb = self.speaker_encoder(reference_audio, sample_rate)

        # コサイン類似度
        similarity = self.speaker_encoder.compute_similarity(gen_emb, ref_emb)

        return similarity

    def forward(
        self,
        base_loss: torch.Tensor,
        generated_audio: Optional[torch.Tensor] = None,
        reference_audio: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
        sample_rate: int = 24000,
    ) -> tuple[torch.Tensor, dict]:
        """TCO重み付き損失を計算

        Args:
            base_loss: ベース損失（Flow Matching損失）[B] または スカラー
            generated_audio: 生成音声 [B, T]（reward未指定時に必要）
            reference_audio: 参照音声 [B, T]（reward未指定時に必要）
            reward: 事前計算された報酬 [B]（指定時は音声から計算しない）
            sample_rate: サンプリングレート

        Returns:
            (重み付き損失, メトリクス辞書)
        """
        if not self.config.enabled:
            return base_loss, {"tco_enabled": False}

        # 報酬を計算または使用
        if reward is None:
            if generated_audio is None or reference_audio is None:
                raise ValueError(
                    "reward が指定されていない場合は generated_audio と "
                    "reference_audio を指定してください"
                )
            with torch.no_grad():
                reward = self.compute_reward(generated_audio, reference_audio, sample_rate)

        # 重みを計算
        weight = self.weight_computer(reward, update_baseline=self.training)

        # 損失の形状を確認
        if base_loss.dim() == 0:
            # スカラー損失の場合
            weighted_loss = base_loss * weight.mean()
        else:
            # バッチ損失の場合
            weighted_loss = (base_loss * weight).mean()

        # メトリクス
        metrics = {
            "tco_enabled": True,
            "tco_reward_mean": reward.mean().item(),
            "tco_reward_std": reward.std().item() if reward.numel() > 1 else 0.0,
            "tco_baseline": self.weight_computer.baseline.item(),
            "tco_weight_mean": weight.mean().item(),
            "tco_weight_std": weight.std().item() if weight.numel() > 1 else 0.0,
        }

        return weighted_loss, metrics


class TCOTrainingMixin:
    """TCO訓練のためのMixinクラス

    既存のTrainerクラスに追加してTCO機能を有効にする。

    Usage:
        >>> class MyTrainer(TCOTrainingMixin, BaseTrainer):
        ...     pass
    """

    def init_tco(
        self,
        config: Optional[TCOConfig] = None,
        speaker_encoder: Optional[nn.Module] = None,
    ) -> None:
        """TCOを初期化

        Args:
            config: TCO設定
            speaker_encoder: 話者エンコーダー
        """
        self.tco_config = config or TCOConfig()
        self.tco_loss = TCOLoss(speaker_encoder, config)

    def apply_tco_weight(
        self,
        loss: torch.Tensor,
        generated_audio: torch.Tensor,
        reference_audio: torch.Tensor,
        sample_rate: int = 24000,
    ) -> tuple[torch.Tensor, dict]:
        """損失にTCO重みを適用

        Args:
            loss: ベース損失
            generated_audio: 生成音声
            reference_audio: 参照音声
            sample_rate: サンプリングレート

        Returns:
            (重み付き損失, メトリクス)
        """
        if not hasattr(self, "tco_loss"):
            return loss, {}

        return self.tco_loss(
            loss,
            generated_audio=generated_audio,
            reference_audio=reference_audio,
            sample_rate=sample_rate,
        )


def create_tco_loss(
    config: Optional[TCOConfig] = None,
    speaker_encoder: Optional[nn.Module] = None,
) -> TCOLoss:
    """TCOLossインスタンスを作成

    Args:
        config: TCO設定
        speaker_encoder: 話者エンコーダー

    Returns:
        TCOLossインスタンス
    """
    return TCOLoss(speaker_encoder, config)
