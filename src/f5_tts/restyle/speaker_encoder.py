"""
Speaker Encoder for TCO (Timbre Consistency Optimization)

WavLMを使用した話者埋め込み抽出。TCOの話者類似度報酬計算に使用。

Reference: ReStyle-TTS (arXiv:2601.03632) Section 3.4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpeakerEncoderConfig:
    """Speaker Encoderの設定

    Attributes:
        model_name: 使用するWavLMモデル名
        embedding_dim: 出力埋め込み次元（Noneの場合はモデルのデフォルト）
        pooling: プーリング方式 ("mean", "first", "last")
        normalize: 埋め込みを正規化するか
        device: 使用するデバイス
    """

    model_name: str = "microsoft/wavlm-base-plus-sv"
    embedding_dim: Optional[int] = None
    pooling: str = "mean"
    normalize: bool = True
    device: Optional[str] = None


class SpeakerEncoder(nn.Module):
    """WavLMベースの話者エンコーダー

    音声から話者埋め込みを抽出する。

    Usage:
        >>> encoder = SpeakerEncoder()
        >>> embedding = encoder(waveform)  # [B, embedding_dim]
        >>> similarity = encoder.compute_similarity(emb1, emb2)  # [B]
    """

    def __init__(self, config: Optional[SpeakerEncoderConfig] = None):
        """
        Args:
            config: エンコーダー設定
        """
        super().__init__()
        self.config = config or SpeakerEncoderConfig()
        self._model = None
        self._processor = None
        self._device = None

    def _ensure_model_loaded(self) -> None:
        """モデルを遅延読み込み"""
        if self._model is not None:
            return

        from transformers import WavLMModel, Wav2Vec2FeatureExtractor

        self._model = WavLMModel.from_pretrained(self.config.model_name)
        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(self.config.model_name)

        # デバイス設定
        if self.config.device:
            self._device = torch.device(self.config.device)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = self._model.to(self._device)
        self._model.eval()

        # 勾配を無効化
        for param in self._model.parameters():
            param.requires_grad = False

    @property
    def device(self) -> torch.device:
        """モデルのデバイスを取得"""
        self._ensure_model_loaded()
        return self._device

    @property
    def embedding_dim(self) -> int:
        """埋め込み次元を取得"""
        self._ensure_model_loaded()
        if self.config.embedding_dim:
            return self.config.embedding_dim
        return self._model.config.hidden_size

    def _pool(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """隠れ状態をプーリング

        Args:
            hidden_states: [B, T, D] 隠れ状態
            attention_mask: [B, T] アテンションマスク

        Returns:
            プーリングされた埋め込み [B, D]
        """
        if self.config.pooling == "first":
            return hidden_states[:, 0]
        elif self.config.pooling == "last":
            if attention_mask is not None:
                # 各バッチの最後の有効なトークンを取得
                lengths = attention_mask.sum(dim=1).long() - 1
                batch_size = hidden_states.size(0)
                return hidden_states[torch.arange(batch_size, device=hidden_states.device), lengths]
            return hidden_states[:, -1]
        else:  # mean
            if attention_mask is not None:
                # マスクされた平均
                mask = attention_mask.unsqueeze(-1).float()
                return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
            return hidden_states.mean(dim=1)

    @torch.no_grad()
    def forward(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """音声から話者埋め込みを抽出

        Args:
            waveform: 音声波形 [B, T] または [T]
            sample_rate: サンプリングレート（16000が期待される）

        Returns:
            話者埋め込み [B, embedding_dim]
        """
        self._ensure_model_loaded()

        # バッチ次元を追加
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # サンプリングレートの確認
        if sample_rate != 16000:
            # リサンプリングが必要な場合
            import torchaudio.functional as AF
            waveform = AF.resample(waveform, sample_rate, 16000)

        # 前処理
        inputs = self._processor(
            waveform.cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        input_values = inputs.input_values.to(self._device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)

        # モデル推論
        outputs = self._model(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )

        # プーリング
        embeddings = self._pool(outputs.last_hidden_state, attention_mask)

        # 正規化
        if self.config.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def compute_similarity(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
    ) -> torch.Tensor:
        """2つの埋め込み間のコサイン類似度を計算

        Args:
            embedding1: 話者埋め込み1 [B, D] または [D]
            embedding2: 話者埋め込み2 [B, D] または [D]

        Returns:
            コサイン類似度 [B] または スカラー
        """
        # 次元を揃える
        if embedding1.dim() == 1:
            embedding1 = embedding1.unsqueeze(0)
        if embedding2.dim() == 1:
            embedding2 = embedding2.unsqueeze(0)

        # 正規化されていない場合は正規化
        if not self.config.normalize:
            embedding1 = F.normalize(embedding1, p=2, dim=-1)
            embedding2 = F.normalize(embedding2, p=2, dim=-1)

        # コサイン類似度
        similarity = (embedding1 * embedding2).sum(dim=-1)

        return similarity.squeeze()

    def encode_audio(
        self,
        audio_path: str,
    ) -> torch.Tensor:
        """音声ファイルから話者埋め込みを抽出

        Args:
            audio_path: 音声ファイルのパス

        Returns:
            話者埋め込み [embedding_dim]
        """
        import torchaudio

        waveform, sample_rate = torchaudio.load(audio_path)

        # モノラルに変換
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        embedding = self(waveform.squeeze(0), sample_rate)
        return embedding.squeeze(0)


def compute_speaker_similarity(
    audio1: torch.Tensor,
    audio2: torch.Tensor,
    encoder: Optional[SpeakerEncoder] = None,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """2つの音声間の話者類似度を計算

    Args:
        audio1: 音声1 [B, T] または [T]
        audio2: 音声2 [B, T] または [T]
        encoder: 使用するエンコーダー（Noneの場合は新規作成）
        sample_rate: サンプリングレート

    Returns:
        話者類似度 [B] または スカラー
    """
    if encoder is None:
        encoder = SpeakerEncoder()

    emb1 = encoder(audio1, sample_rate)
    emb2 = encoder(audio2, sample_rate)

    return encoder.compute_similarity(emb1, emb2)
