"""
ReStyle-TTS: Relative and Continuous Style Control for Zero-Shot Speech Synthesis

This module implements ReStyle-TTS features for F5-TTS:
- DCFG (Decoupled Classifier-Free Guidance) ✅
- Style LoRA ✅
- OLoRA Fusion ✅
- TCO (Timbre Consistency Optimization) 📋

Reference: arXiv:2601.03632
"""

from f5_tts.restyle.dcfg import DCFGConfig, dcfg_combine
from f5_tts.restyle.style_lora import (
    STYLE_ATTRIBUTES,
    STYLE_CATEGORIES,
    StyleLoRAConfig,
    StyleLoRAManager,
    create_lora_model,
    freeze_base_model,
    unfreeze_lora_params,
    count_trainable_params,
)
from f5_tts.restyle.olora_fusion import (
    OLoRAConfig,
    OLoRAFusion,
    fuse_lora_weights,
    orthogonalize_loras,
    compute_orthogonal_projection,
)

__all__ = [
    # DCFG
    "DCFGConfig",
    "dcfg_combine",
    # Style LoRA
    "STYLE_ATTRIBUTES",
    "STYLE_CATEGORIES",
    "StyleLoRAConfig",
    "StyleLoRAManager",
    "create_lora_model",
    "freeze_base_model",
    "unfreeze_lora_params",
    "count_trainable_params",
    # OLoRA Fusion
    "OLoRAConfig",
    "OLoRAFusion",
    "fuse_lora_weights",
    "orthogonalize_loras",
    "compute_orthogonal_projection",
]
