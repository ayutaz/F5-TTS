"""
OLoRA Fusion (Orthogonal LoRA Fusion) for ReStyle-TTS

複数のStyle LoRAを直交射影により干渉なく融合する。

Reference: ReStyle-TTS (arXiv:2601.03632) Section 3.3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class OLoRAConfig:
    """OLoRA Fusion の設定

    Attributes:
        orthogonalize: 直交化を有効にするか（デフォルト: True）
        epsilon: 数値安定性のための小さな値（デフォルト: 1e-8）
        use_svd: SVDベースの直交化を使用するか（デフォルト: False）
    """

    orthogonalize: bool = True
    epsilon: float = 1e-8
    use_svd: bool = False


def compute_orthogonal_projection(
    vectors: torch.Tensor,
    target_idx: int,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """指定したベクトルを他のベクトルの部分空間に直交射影

    v̂_i = (I - P_{-i}) @ v_i
    where P_{-i} = V_{-i}^T @ pinv(V_{-i}^T)

    Args:
        vectors: [N, D] 形状のベクトル群（N個のベクトル、各D次元）
        target_idx: 直交化するベクトルのインデックス
        epsilon: 数値安定性のための小さな値

    Returns:
        直交化されたベクトル [D]
    """
    N, D = vectors.shape

    if N == 1:
        # ベクトルが1つの場合は直交化不要
        return vectors[0]

    # target_idx 以外のベクトルを取得
    mask = torch.ones(N, dtype=torch.bool, device=vectors.device)
    mask[target_idx] = False
    V_minus_i = vectors[mask]  # [N-1, D]

    # ターゲットベクトル
    v_i = vectors[target_idx]  # [D]

    # 射影行列を計算: P_{-i} = V_{-i}^T @ pinv(V_{-i}^T)
    # V_{-i}^T: [D, N-1]
    V_minus_i_T = V_minus_i.T

    # pinv(V_{-i}^T): [N-1, D]
    try:
        pinv_V = torch.linalg.pinv(V_minus_i_T)
    except RuntimeError:
        # pinvが失敗した場合は元のベクトルを返す
        return v_i

    # P_{-i} @ v_i
    # [D, N-1] @ [N-1, D] @ [D] = [D]
    projected = V_minus_i_T @ (pinv_V @ v_i)

    # 直交成分: v̂_i = v_i - P_{-i} @ v_i
    v_orth = v_i - projected

    return v_orth


def orthogonalize_loras(
    lora_deltas: Dict[str, torch.Tensor],
    epsilon: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """複数のLoRA差分を直交化

    Args:
        lora_deltas: {LoRA名: 差分テンソル [D]} の辞書
        epsilon: 数値安定性のための小さな値

    Returns:
        直交化されたLoRA差分の辞書
    """
    if len(lora_deltas) <= 1:
        # 1つ以下なら直交化不要
        return lora_deltas

    names = list(lora_deltas.keys())
    vectors = torch.stack([lora_deltas[name] for name in names])  # [N, D]

    orthogonalized = {}
    for i, name in enumerate(names):
        v_orth = compute_orthogonal_projection(vectors, i, epsilon)
        orthogonalized[name] = v_orth

    return orthogonalized


def fuse_lora_weights(
    lora_state_dicts: Dict[str, Dict[str, torch.Tensor]],
    alphas: Dict[str, float],
    orthogonalize: bool = True,
    epsilon: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """複数のLoRA重みを融合

    OLoRA式:
    ΔW_fuse = Σ α_i * ΔŴ_i

    Args:
        lora_state_dicts: {LoRA名: state_dict} の辞書
        alphas: {LoRA名: 強度} の辞書
        orthogonalize: 直交化を有効にするか
        epsilon: 数値安定性のための小さな値

    Returns:
        融合されたLoRA state_dict
    """
    if not lora_state_dicts:
        return {}

    # 有効なLoRA（alpha > 0）のみフィルタ
    active_loras = {
        name: state_dict
        for name, state_dict in lora_state_dicts.items()
        if name in alphas and abs(alphas[name]) > 1e-6
    }

    if not active_loras:
        return {}

    if len(active_loras) == 1:
        # 1つのLoRAのみの場合
        name = list(active_loras.keys())[0]
        alpha = alphas[name]
        return {
            key: alpha * value
            for key, value in active_loras[name].items()
        }

    # 全てのキーを収集
    all_keys = set()
    for state_dict in active_loras.values():
        all_keys.update(state_dict.keys())

    # キーごとに融合
    fused_state = {}
    for key in all_keys:
        # このキーを持つLoRAを収集
        key_loras = {
            name: state_dict[key]
            for name, state_dict in active_loras.items()
            if key in state_dict
        }

        if not key_loras:
            continue

        if len(key_loras) == 1:
            # 1つのLoRAのみがこのキーを持つ場合
            name = list(key_loras.keys())[0]
            fused_state[key] = alphas[name] * key_loras[name]
            continue

        # 形状を取得
        first_tensor = list(key_loras.values())[0]
        original_shape = first_tensor.shape
        device = first_tensor.device
        dtype = first_tensor.dtype

        # フラット化
        flat_loras = {
            name: tensor.flatten().to(dtype=torch.float32)
            for name, tensor in key_loras.items()
        }

        if orthogonalize:
            # 直交化
            flat_loras = orthogonalize_loras(flat_loras, epsilon)

        # 重み付き合成
        fused = torch.zeros_like(list(flat_loras.values())[0])
        for name, flat_tensor in flat_loras.items():
            fused = fused + alphas[name] * flat_tensor

        # 元の形状とdtypeに戻す
        fused_state[key] = fused.reshape(original_shape).to(dtype=dtype)

    return fused_state


class OLoRAFusion:
    """OLoRA Fusion クラス

    複数のStyle LoRAを直交融合で組み合わせる。

    Usage:
        >>> fusion = OLoRAFusion()
        >>> fusion.add_lora("pitch_high", pitch_high_state_dict)
        >>> fusion.add_lora("angry", angry_state_dict)
        >>> fused = fusion.fuse({"pitch_high": 1.0, "angry": 0.5})
    """

    def __init__(self, config: Optional[OLoRAConfig] = None):
        """
        Args:
            config: OLoRA設定
        """
        self.config = config or OLoRAConfig()
        self.lora_state_dicts: Dict[str, Dict[str, torch.Tensor]] = {}

    def add_lora(
        self,
        name: str,
        state_dict: Dict[str, torch.Tensor],
    ) -> None:
        """LoRAを追加

        Args:
            name: LoRA名
            state_dict: LoRAの state_dict
        """
        self.lora_state_dicts[name] = state_dict

    def remove_lora(self, name: str) -> None:
        """LoRAを削除

        Args:
            name: LoRA名
        """
        if name in self.lora_state_dicts:
            del self.lora_state_dicts[name]

    def clear(self) -> None:
        """全てのLoRAを削除"""
        self.lora_state_dicts.clear()

    def get_lora_names(self) -> List[str]:
        """登録されているLoRA名のリストを取得"""
        return list(self.lora_state_dicts.keys())

    def fuse(
        self,
        alphas: Dict[str, float],
        orthogonalize: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """LoRAを融合

        Args:
            alphas: {LoRA名: 強度} の辞書
            orthogonalize: 直交化を有効にするか（Noneの場合はconfigの値を使用）

        Returns:
            融合されたLoRA state_dict
        """
        if orthogonalize is None:
            orthogonalize = self.config.orthogonalize

        return fuse_lora_weights(
            self.lora_state_dicts,
            alphas,
            orthogonalize=orthogonalize,
            epsilon=self.config.epsilon,
        )

    def compute_interference(
        self,
        lora1_name: str,
        lora2_name: str,
    ) -> float:
        """2つのLoRA間の干渉度を計算

        干渉度はコサイン類似度の絶対値として計算。
        0に近いほど直交（干渉が少ない）。

        Args:
            lora1_name: 1つ目のLoRA名
            lora2_name: 2つ目のLoRA名

        Returns:
            干渉度 [0, 1]
        """
        if lora1_name not in self.lora_state_dicts:
            raise ValueError(f"LoRA '{lora1_name}' が見つかりません")
        if lora2_name not in self.lora_state_dicts:
            raise ValueError(f"LoRA '{lora2_name}' が見つかりません")

        state1 = self.lora_state_dicts[lora1_name]
        state2 = self.lora_state_dicts[lora2_name]

        # 共通のキーのみ使用
        common_keys = set(state1.keys()) & set(state2.keys())
        if not common_keys:
            return 0.0

        # 全ての重みを連結してフラット化
        flat1 = torch.cat([state1[k].flatten() for k in sorted(common_keys)])
        flat2 = torch.cat([state2[k].flatten() for k in sorted(common_keys)])

        # コサイン類似度
        cos_sim = torch.nn.functional.cosine_similarity(
            flat1.unsqueeze(0).float(),
            flat2.unsqueeze(0).float(),
        )

        return abs(cos_sim.item())

    def get_interference_matrix(self) -> Dict[str, Dict[str, float]]:
        """全てのLoRAペア間の干渉行列を計算

        Returns:
            {LoRA名: {LoRA名: 干渉度}} の辞書
        """
        names = self.get_lora_names()
        matrix = {}

        for name1 in names:
            matrix[name1] = {}
            for name2 in names:
                if name1 == name2:
                    matrix[name1][name2] = 1.0
                else:
                    matrix[name1][name2] = self.compute_interference(name1, name2)

        return matrix
