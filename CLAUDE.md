# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is F5-TTS extended with **ReStyle-TTS** capabilities for continuous and reference-relative style control in zero-shot speech synthesis.

Based on:
- F5-TTS: Flow matching based TTS (original repository)
- ReStyle-TTS: arXiv:2601.03632 - Relative and Continuous Style Control

## Implementation Status

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | DCFG (Decoupled CFG) | ✅ 完了 |
| 2 | Style LoRA | ✅ 完了 |
| 3 | OLoRA Fusion | ✅ 完了 |
| 4 | TCO | ✅ 完了 |
| 5 | 推論インターフェース | ✅ 完了 |
| 6 | モデル公開 | ✅ 完了 |

詳細は `docs/IMPLEMENTATION_PLAN.md` を参照。

## Build & Run Commands

```bash
# Install dependencies (using uv)
uv sync

# Or using pip
pip install -e .

# Run tests
uv run pytest tests/ -v

# Training (Hydra config)
python src/f5_tts/train/train.py --config-name F5TTS_v1_Base

# Style LoRA training
python -m f5_tts.train.train_style_lora --config-name ReStyleTTS_Base \
    style_attribute=pitch_high \
    pretrained_checkpoint=path/to/base_model.pt

# Inference CLI
python src/f5_tts/infer/infer_cli.py \
    --model F5TTS_v1_Base \
    --ref_audio path/to/ref.wav \
    --ref_text "reference text" \
    --gen_text "text to generate"

# Gradio UI
python src/f5_tts/infer/infer_gradio.py
```

## Architecture Overview

### F5-TTS Base Model
```
Input: Reference Audio + Text
    │
    ▼
┌─────────────────────────────┐
│  Mel Spectrogram Extraction │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  DiT (Diffusion Transformer)│  ← 22 layers, dim=1024
│  - TextEmbedding            │
│  - InputEmbedding           │
│  - DiTBlock × 22            │
│    - AdaLayerNorm           │
│    - Attention (Q,K,V,Out)  │
│    - FeedForward            │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Flow Matching (ODE Solver) │  ← torchdiffeq.odeint
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Vocoder (Vocos/BigVGAN)    │
└─────────────────────────────┘
    │
    ▼
Output: Generated Audio
```

### ReStyle-TTS Extensions

1. **DCFG (Decoupled CFG)** ✅: Separate text/reference guidance
2. **Style LoRA** ✅: Attribute-specific adapters (pitch, energy, emotions)
3. **OLoRA Fusion** ✅: Orthogonal multi-LoRA composition
4. **TCO** ✅: Timbre consistency optimization with speaker similarity reward

## Key Files

### Core Model
- `src/f5_tts/model/cfm.py` - CFM class, sample() with CFG/DCFG
- `src/f5_tts/model/backbones/dit.py` - DiT transformer (dcfg_infer対応済み)
- `src/f5_tts/model/modules.py` - Attention, FeedForward, DiTBlock
- `src/f5_tts/model/trainer.py` - Training loop

### ReStyle-TTS Extensions ✅
- `src/f5_tts/restyle/__init__.py` - ReStyleモジュール
- `src/f5_tts/restyle/dcfg.py` - DCFG実装 (DCFGConfig, dcfg_combine)
- `src/f5_tts/restyle/style_lora.py` - Style LoRA管理 (StyleLoRAManager + OLoRA統合)
- `src/f5_tts/restyle/olora_fusion.py` - OLoRA直交融合 (OLoRAFusion, fuse_lora_weights)
- `src/f5_tts/restyle/speaker_encoder.py` - WavLM話者エンコーダー (SpeakerEncoder)
- `src/f5_tts/restyle/tco.py` - TCO損失 (TCOLoss, TCOWeightComputer)
- `src/f5_tts/train/train_style_lora.py` - LoRA訓練スクリプト

### Inference
- `src/f5_tts/infer/utils_infer.py` - Inference utilities
- `src/f5_tts/api.py` - High-level API

### Config
- `src/f5_tts/configs/ReStyleTTS_Base.yaml` - ReStyle-TTS設定
- `src/f5_tts/configs/*.yaml` - その他Hydra configs

### Tests
- `tests/test_dcfg.py` - DCFGテスト (16テスト)
- `tests/test_style_lora.py` - Style LoRAテスト (21テスト)
- `tests/test_olora_fusion.py` - OLoRA Fusionテスト (30テスト)
- `tests/test_tco.py` - TCOテスト (31テスト)

## DCFG Implementation ✅

Location: `cfm.py`, `dit.py`, `restyle/dcfg.py`

```python
# DCFG formula (implemented)
f̂ = f_{∅,t} + λ_t(f_{∅,t} - f_{∅,∅}) + λ_a(f_{a,t} - f_{∅,t})
# Requires 3 forward passes: uncond, text-only, full-cond
# Default: λ_t=2.0, λ_a=0.5

# Usage
from f5_tts.model.cfm import CFM

output, _ = model.sample(
    cond, text, duration,
    use_dcfg=True,
    lambda_t=2.0,
    lambda_a=0.5,
)
```

## Style LoRA Implementation ✅

Location: `restyle/style_lora.py`, `train/train_style_lora.py`

```python
# Style LoRA config
lora_config = {
    "rank": 32,
    "alpha": 64,
    "target_modules": ["to_q", "to_k", "to_v", "to_out.0", "ff.ff.0", "ff.ff.2"],
    "dropout": 0.0,
}

# Available style attributes
STYLE_ATTRIBUTES = {
    "pitch_high", "pitch_low",       # ピッチ
    "energy_high", "energy_low",     # エネルギー
    "angry", "happy", "sad",         # 感情
    "fear", "disgusted", "surprised"
}

# Usage
from f5_tts.restyle import StyleLoRAManager

manager = StyleLoRAManager(model.transformer)
manager.load_lora("pitch_high", "path/to/pitch_high.safetensors")

with manager.apply_styles({"pitch_high": 1.0}):
    output = model.sample(...)
```

## OLoRA Fusion Implementation ✅

Location: `restyle/olora_fusion.py`, `restyle/style_lora.py`

```python
# Orthogonal projection formula
v̂_i = (I - P_{-i}) @ v_i  # where P_{-i} = V_{-i}^T @ pinv(V_{-i}^T)
ΔW_fuse = Σ α_i * ΔŴ_i   # weighted sum of orthogonalized LoRAs

# Usage with StyleLoRAManager
from f5_tts.restyle import StyleLoRAManager, OLoRAConfig

manager = StyleLoRAManager(model.transformer, olora_config=OLoRAConfig())
manager.load_lora("pitch_high", "path/to/pitch_high.safetensors")
manager.load_lora("angry", "path/to/angry.safetensors")

# Apply multiple styles with OLoRA fusion (default)
with manager.apply_styles({"pitch_high": 1.0, "angry": 0.5}, use_olora=True):
    output = model.sample(...)

# Low-level API
from f5_tts.restyle import OLoRAFusion

fusion = OLoRAFusion()
fusion.add_lora("pitch_high", pitch_high_state_dict)
fusion.add_lora("angry", angry_state_dict)
interference = fusion.compute_interference("pitch_high", "angry")  # 0.0-1.0
fused = fusion.fuse({"pitch_high": 1.0, "angry": 0.5})
```

## TCO Implementation ✅

Location: `restyle/tco.py`, `restyle/speaker_encoder.py`

```python
# Advantage-weighted flow matching loss formula
w_t = 1 + λ * tanh(β * A_t)
A_t = r_t - b_t  # advantage = reward - baseline
L_total = w_t * L_FM
# λ=0.2, β=5.0, μ=0.9 (EMA baseline)

# Usage
from f5_tts.restyle import TCOLoss, TCOConfig, SpeakerEncoder

# Create TCO loss
config = TCOConfig(lambda_reward=0.2, beta=5.0, mu=0.9)
tco_loss = TCOLoss(config=config)

# In training loop
base_loss = compute_flow_matching_loss(...)
weighted_loss, metrics = tco_loss(
    base_loss,
    generated_audio=gen_audio,
    reference_audio=ref_audio,
)

# Or with pre-computed reward
from f5_tts.restyle import compute_speaker_similarity
reward = compute_speaker_similarity(gen_audio, ref_audio)
weighted_loss, metrics = tco_loss(base_loss, reward=reward)

# Metrics include:
# - tco_reward_mean: average speaker similarity
# - tco_baseline: EMA baseline
# - tco_weight_mean: average loss weight
```

## Development Notes

- Use `uv` for dependency management (`uv sync`, `uv add`)
- Use `accelerate` for multi-GPU training
- EMA model is used for inference
- Checkpoints saved as `.safetensors` or `.pt`
- Audio: 24kHz, 100 mel channels, hop_length=256
- Python version: >=3.10, <3.13 (due to av package compatibility)
- Tests: `uv run pytest tests/ -v`

## Trained Style LoRA Models

学習済みStyle LoRAモデルはHugging Face Hubで公開:
- **Repository**: [ayousanz/restyle-tts-style-loras](https://huggingface.co/ayousanz/restyle-tts-style-loras)

| Adapter | Description |
|---------|-------------|
| `pitch_high` | 高いピッチ |
| `pitch_low` | 低いピッチ |
| `energy_high` | 高いエネルギー |
| `energy_low` | 低いエネルギー |

```python
from huggingface_hub import hf_hub_download

# Download LoRA
lora_path = hf_hub_download(
    repo_id="ayousanz/restyle-tts-style-loras",
    filename="loras/pitch_high.safetensors"
)
```

## Documentation

- `docs/ROADMAP.md` - 実装ロードマップ
- `docs/IMPLEMENTATION_PLAN.md` - 詳細実装計画
- `docs/RESTYLE_TTS_ARCHITECTURE.md` - アーキテクチャ詳細
