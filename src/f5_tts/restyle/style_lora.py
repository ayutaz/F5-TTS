"""
Style LoRA management for ReStyle-TTS

各スタイル属性（ピッチ、エネルギー、感情）に特化したLoRAアダプターを管理する。

Reference: ReStyle-TTS (arXiv:2601.03632) Section 3.2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel, set_peft_model_state_dict

if TYPE_CHECKING:
    from f5_tts.model.backbones.dit import DiT


# スタイル属性の定義
STYLE_ATTRIBUTES = {
    # ピッチ
    "pitch_high": "高いピッチ",
    "pitch_low": "低いピッチ",
    # エネルギー
    "energy_high": "高いエネルギー",
    "energy_low": "低いエネルギー",
    # 感情（Ekmanの基本6感情）
    "angry": "怒り",
    "happy": "喜び",
    "sad": "悲しみ",
    "fear": "恐怖",
    "disgusted": "嫌悪",
    "surprised": "驚き",
}

# スタイル属性のカテゴリ
STYLE_CATEGORIES = {
    "pitch": ["pitch_high", "pitch_low"],
    "energy": ["energy_high", "energy_low"],
    "emotion": ["angry", "happy", "sad", "fear", "disgusted", "surprised"],
}

# デフォルトのLoRAターゲットモジュール（DiTの線形層）
DEFAULT_TARGET_MODULES = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "ff.ff.0",
    "ff.ff.2",
]


@dataclass
class StyleLoRAConfig:
    """Style LoRAの設定

    Attributes:
        rank: LoRAのランク（デフォルト: 32）
        alpha: LoRAのアルファ値（デフォルト: 64）
        target_modules: LoRAを適用するモジュール名のリスト
        dropout: ドロップアウト率（デフォルト: 0.0）
        bias: バイアスの扱い（デフォルト: "none"）
    """

    rank: int = 32
    alpha: int = 64
    target_modules: list[str] = field(default_factory=lambda: DEFAULT_TARGET_MODULES.copy())
    dropout: float = 0.0
    bias: str = "none"

    def to_lora_config(self) -> LoraConfig:
        """peftのLoraConfigに変換"""
        return LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            target_modules=self.target_modules,
            lora_dropout=self.dropout,
            bias=self.bias,
        )


class StyleLoRAManager:
    """Style LoRAの管理クラス

    複数のスタイル属性に対応したLoRAアダプターを管理する。

    Usage:
        >>> manager = StyleLoRAManager(base_model)
        >>> manager.load_lora("pitch_high", "path/to/pitch_high.safetensors")
        >>> manager.load_lora("angry", "path/to/angry.safetensors")
        >>> # スタイルを適用して推論
        >>> with manager.apply_styles({"pitch_high": 1.0, "angry": 0.5}):
        ...     output = model.sample(...)
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: StyleLoRAConfig | None = None,
    ):
        """
        Args:
            base_model: ベースモデル（DiT）
            config: LoRA設定（Noneの場合はデフォルト設定を使用）
        """
        self.base_model = base_model
        self.config = config or StyleLoRAConfig()
        self.loaded_loras: dict[str, dict[str, torch.Tensor]] = {}
        self._original_state: dict[str, torch.Tensor] | None = None
        self._peft_model: PeftModel | None = None

    def _ensure_peft_model(self) -> PeftModel:
        """PEFTモデルを初期化（まだの場合）"""
        if self._peft_model is None:
            lora_config = self.config.to_lora_config()
            self._peft_model = get_peft_model(self.base_model, lora_config)
        return self._peft_model

    def load_lora(
        self,
        style_name: str,
        checkpoint_path: str | Path,
    ) -> None:
        """LoRAチェックポイントを読み込み

        Args:
            style_name: スタイル属性名（例: "pitch_high", "angry"）
            checkpoint_path: チェックポイントファイルのパス

        Raises:
            ValueError: 不明なスタイル属性名の場合
            FileNotFoundError: チェックポイントが存在しない場合
        """
        if style_name not in STYLE_ATTRIBUTES:
            raise ValueError(
                f"不明なスタイル属性: {style_name}. "
                f"利用可能な属性: {list(STYLE_ATTRIBUTES.keys())}"
            )

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"チェックポイントが見つかりません: {checkpoint_path}")

        # チェックポイントを読み込み
        if checkpoint_path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        self.loaded_loras[style_name] = state_dict

    def unload_lora(self, style_name: str) -> None:
        """LoRAをアンロード

        Args:
            style_name: スタイル属性名
        """
        if style_name in self.loaded_loras:
            del self.loaded_loras[style_name]

    def unload_all(self) -> None:
        """全てのLoRAをアンロード"""
        self.loaded_loras.clear()

    def get_loaded_styles(self) -> list[str]:
        """読み込み済みのスタイル属性名を取得"""
        return list(self.loaded_loras.keys())

    def _apply_lora_weights(
        self,
        style_weights: dict[str, float],
    ) -> None:
        """LoRA重みを適用

        Args:
            style_weights: {スタイル名: 強度} の辞書
        """
        if not style_weights:
            return

        # 有効なスタイルのみフィルタ
        active_styles = {
            name: weight
            for name, weight in style_weights.items()
            if name in self.loaded_loras and abs(weight) > 1e-6
        }

        if not active_styles:
            return

        # PEFTモデルを確保
        peft_model = self._ensure_peft_model()

        # 複数のLoRAを重み付けで合成
        combined_state = {}
        for style_name, weight in active_styles.items():
            lora_state = self.loaded_loras[style_name]
            for key, value in lora_state.items():
                if key in combined_state:
                    combined_state[key] = combined_state[key] + weight * value
                else:
                    combined_state[key] = weight * value.clone()

        # 合成した重みを適用
        set_peft_model_state_dict(peft_model, combined_state)

    def _restore_original_weights(self) -> None:
        """元の重みを復元"""
        if self._peft_model is not None:
            # LoRA重みをゼロにリセット
            for name, param in self._peft_model.named_parameters():
                if "lora_" in name:
                    param.data.zero_()

    class _StyleContext:
        """スタイル適用のコンテキストマネージャー"""

        def __init__(self, manager: "StyleLoRAManager", style_weights: dict[str, float]):
            self.manager = manager
            self.style_weights = style_weights

        def __enter__(self):
            self.manager._apply_lora_weights(self.style_weights)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.manager._restore_original_weights()
            return False

    def apply_styles(self, style_weights: dict[str, float]) -> _StyleContext:
        """スタイルを適用するコンテキストマネージャー

        Args:
            style_weights: {スタイル名: 強度} の辞書
                例: {"pitch_high": 1.0, "angry": 0.5}

        Returns:
            コンテキストマネージャー

        Usage:
            >>> with manager.apply_styles({"pitch_high": 1.0}):
            ...     output = model.sample(...)
        """
        return self._StyleContext(self, style_weights)

    def get_style_info(self, style_name: str) -> dict:
        """スタイル属性の情報を取得

        Args:
            style_name: スタイル属性名

        Returns:
            スタイル情報の辞書
        """
        if style_name not in STYLE_ATTRIBUTES:
            raise ValueError(f"不明なスタイル属性: {style_name}")

        # カテゴリを特定
        category = None
        for cat, styles in STYLE_CATEGORIES.items():
            if style_name in styles:
                category = cat
                break

        return {
            "name": style_name,
            "description": STYLE_ATTRIBUTES[style_name],
            "category": category,
            "loaded": style_name in self.loaded_loras,
        }


def create_lora_model(
    base_model: nn.Module,
    config: StyleLoRAConfig | None = None,
) -> PeftModel:
    """LoRAアダプターを追加したモデルを作成

    訓練用にベースモデルにLoRAアダプターを追加する。

    Args:
        base_model: ベースモデル（DiT）
        config: LoRA設定

    Returns:
        LoRAアダプターが追加されたPeftModel
    """
    config = config or StyleLoRAConfig()
    lora_config = config.to_lora_config()
    return get_peft_model(base_model, lora_config)


def freeze_base_model(model: nn.Module) -> None:
    """ベースモデルのパラメータを凍結

    LoRA訓練時にベースモデルの重みを固定する。

    Args:
        model: 凍結するモデル
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_lora_params(model: PeftModel) -> None:
    """LoRAパラメータのみを訓練可能にする

    Args:
        model: PeftModel
    """
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    """訓練可能なパラメータ数をカウント

    Args:
        model: モデル

    Returns:
        (訓練可能パラメータ数, 全パラメータ数)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
