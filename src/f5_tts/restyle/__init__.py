"""
ReStyle-TTS: Relative and Continuous Style Control for Zero-Shot Speech Synthesis

This module implements ReStyle-TTS features for F5-TTS:
- DCFG (Decoupled Classifier-Free Guidance)
- Style LoRA (Phase 2)
- OLoRA Fusion (Phase 3)
- TCO (Timbre Consistency Optimization) (Phase 4)

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
]
