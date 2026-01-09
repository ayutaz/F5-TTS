# ReStyle-TTS 実装計画

## 実装概要

F5-TTSをベースにReStyle-TTSの機能を追加実装する。

**現在の進捗**: Phase 1-5 ✅ 全完了 (DCFG, Style LoRA, OLoRA Fusion, TCO, 推論インターフェース)

## ディレクトリ構造

```
F5-TTS/
├── src/f5_tts/
│   ├── model/
│   │   ├── cfm.py              # ✅ 修正済: DCFG対応
│   │   ├── trainer.py          # (TCOはTCOTrainingMixinで統合可能)
│   │   └── backbones/
│   │       └── dit.py          # ✅ 修正済: 3パス推論対応
│   │
│   ├── restyle/                # ✅ 新規作成
│   │   ├── __init__.py         # ✅ 作成済
│   │   ├── dcfg.py             # ✅ 作成済: DCFG実装
│   │   ├── style_lora.py       # ✅ 作成済: Style LoRA管理 + OLoRA統合
│   │   ├── olora_fusion.py     # ✅ 作成済: OLoRA直交融合
│   │   ├── tco.py              # ✅ 作成済: TCO損失
│   │   └── speaker_encoder.py  # ✅ 作成済: WavLM話者エンコーダー
│   │
│   ├── configs/
│   │   └── ReStyleTTS_Base.yaml  # ✅ 作成済
│   │
│   └── train/
│       └── train_style_lora.py   # ✅ 作成済
│
├── tests/
│   ├── test_dcfg.py            # ✅ 作成済 (14テスト)
│   ├── test_style_lora.py      # ✅ 作成済 (21テスト)
│   ├── test_olora_fusion.py    # ✅ 作成済 (30テスト)
│   └── test_tco.py             # ✅ 作成済 (31テスト)
│
├── docs/
│   ├── ROADMAP.md              # ✅ 作成済
│   ├── RESTYLE_TTS_ARCHITECTURE.md
│   └── IMPLEMENTATION_PLAN.md  # このファイル
│
└── CLAUDE.md
```

---

## Phase 1: DCFG (Decoupled Classifier-Free Guidance) ✅ 完了

### 1.1 目的
テキストと参照音声のガイダンスを分離し、参照スタイルへの依存を減らす。

### 1.2 実装済みファイル

#### `src/f5_tts/restyle/dcfg.py` ✅
```python
from dataclasses import dataclass
import torch

@dataclass
class DCFGConfig:
    lambda_t: float = 2.0   # テキストガイダンス強度
    lambda_a: float = 0.5   # 参照音声ガイダンス強度
    enabled: bool = True

def dcfg_combine(f_full, f_text, f_null, lambda_t=2.0, lambda_a=0.5):
    """DCFG式: f̂ = f_t + λ_t(f_t - f_∅) + λ_a(f_at - f_t)"""
    return f_text + lambda_t * (f_text - f_null) + lambda_a * (f_full - f_text)
```

#### `src/f5_tts/model/backbones/dit.py` ✅ 修正済み
- `forward()` に `dcfg_infer` パラメータ追加
- 3パス推論対応（full_cond, text_only, uncond）

#### `src/f5_tts/model/cfm.py` ✅ 修正済み
- `sample()` に `use_dcfg`, `lambda_t`, `lambda_a` パラメータ追加
- DCFG式による予測合成

### 1.3 使用方法
```python
# 従来CFG（後方互換）
output, _ = model.sample(cond, text, duration, cfg_strength=2.0)

# DCFG（新機能）
output, _ = model.sample(
    cond, text, duration,
    use_dcfg=True,
    lambda_t=2.0,
    lambda_a=0.5,
)
```

---

## Phase 2: Style LoRA ✅ 完了

### 2.1 目的
各スタイル属性に特化したLoRAアダプターを学習し、連続的なスタイル制御を可能にする。

### 2.2 実装済みファイル

#### `src/f5_tts/restyle/style_lora.py` ✅
```python
# スタイル属性
STYLE_ATTRIBUTES = {
    "pitch_high", "pitch_low",       # ピッチ
    "energy_high", "energy_low",     # エネルギー
    "angry", "happy", "sad",         # 感情
    "fear", "disgusted", "surprised"
}

# 設定
@dataclass
class StyleLoRAConfig:
    rank: int = 32
    alpha: int = 64
    target_modules: list = ["to_q", "to_k", "to_v", "to_out.0", "ff.ff.0", "ff.ff.2"]
    dropout: float = 0.0

# 管理クラス
class StyleLoRAManager:
    def load_lora(self, style_name, checkpoint_path): ...
    def apply_styles(self, style_weights): ...  # コンテキストマネージャー
```

#### `src/f5_tts/train/train_style_lora.py` ✅
- Hydra設定ベースの訓練スクリプト
- ベースモデル凍結 + LoRAのみ訓練
- safetensors形式で保存

#### `src/f5_tts/configs/ReStyleTTS_Base.yaml` ✅
```yaml
dcfg:
  enabled: true
  lambda_t: 2.0
  lambda_a: 0.5

lora:
  rank: 32
  alpha: 64
  target_modules: [to_q, to_k, to_v, to_out.0, ff.ff.0, ff.ff.2]
```

### 2.3 使用方法

**訓練:**
```bash
python -m f5_tts.train.train_style_lora --config-name ReStyleTTS_Base \
    style_attribute=pitch_high \
    pretrained_checkpoint=path/to/base_model.pt
```

**推論:**
```python
from f5_tts.restyle import StyleLoRAManager

manager = StyleLoRAManager(model.transformer)
manager.load_lora("pitch_high", "path/to/pitch_high.safetensors")

with manager.apply_styles({"pitch_high": 1.0}):
    output = model.sample(...)
```

---

## Phase 3: OLoRA Fusion ✅ 完了

### 3.1 目的
複数のStyle LoRAを干渉なく同時適用するための直交融合メカニズム。

### 3.2 実装済みファイル

#### `src/f5_tts/restyle/olora_fusion.py` ✅
```python
from dataclasses import dataclass

@dataclass
class OLoRAConfig:
    orthogonalize: bool = True
    epsilon: float = 1e-8
    use_svd: bool = False

class OLoRAFusion:
    def add_lora(self, name, state_dict): ...
    def fuse(self, alphas, orthogonalize=None): ...
    def compute_interference(self, lora1_name, lora2_name): ...
    def get_interference_matrix(self): ...

def fuse_lora_weights(lora_state_dicts, alphas, orthogonalize=True): ...
def orthogonalize_loras(lora_deltas): ...
def compute_orthogonal_projection(vectors, target_idx): ...
```

#### `src/f5_tts/restyle/style_lora.py` ✅ OLoRA統合
```python
class StyleLoRAManager:
    def __init__(self, base_model, config=None, olora_config=None): ...
    def apply_styles(self, style_weights, use_olora=True): ...
```

### 3.3 数式
```
v̂_i = (I - P_{-i}) @ v_i
P_{-i} = V_{-i}^T @ pinv(V_{-i}^T)
ΔW_fuse = Σ α_i * ΔŴ_i
```

### 3.4 使用方法

**StyleLoRAManager（高レベルAPI）:**
```python
from f5_tts.restyle import StyleLoRAManager, OLoRAConfig

manager = StyleLoRAManager(model.transformer, olora_config=OLoRAConfig())
manager.load_lora("pitch_high", "path/to/pitch_high.safetensors")
manager.load_lora("angry", "path/to/angry.safetensors")

# OLoRA有効（デフォルト）
with manager.apply_styles({"pitch_high": 1.0, "angry": 0.5}):
    output = model.sample(...)
```

**OLoRAFusion（低レベルAPI）:**
```python
from f5_tts.restyle import OLoRAFusion

fusion = OLoRAFusion()
fusion.add_lora("pitch_high", pitch_high_state_dict)
fusion.add_lora("angry", angry_state_dict)

# 干渉度を計算
interference = fusion.compute_interference("pitch_high", "angry")

# 融合
fused = fusion.fuse({"pitch_high": 1.0, "angry": 0.5})
```

---

## Phase 4: TCO (Timbre Consistency Optimization) ✅ 完了

### 4.1 目的
DCFGで参照依存を減らした際の音色劣化を補償する。

### 4.2 実装済みファイル

#### `src/f5_tts/restyle/speaker_encoder.py` ✅
```python
@dataclass
class SpeakerEncoderConfig:
    model_name: str = "microsoft/wavlm-base-plus-sv"
    pooling: str = "mean"
    normalize: bool = True

class SpeakerEncoder(nn.Module):
    def forward(self, waveform, sample_rate=16000): ...
    def compute_similarity(self, emb1, emb2): ...
```

#### `src/f5_tts/restyle/tco.py` ✅
```python
@dataclass
class TCOConfig:
    lambda_reward: float = 0.2
    beta: float = 5.0
    mu: float = 0.9

class TCOWeightComputer(nn.Module):
    def compute_advantage(self, reward): ...
    def compute_weight(self, advantage): ...
    def update_baseline(self, reward): ...

class TCOLoss(nn.Module):
    def forward(self, base_loss, generated_audio=None,
                reference_audio=None, reward=None): ...

class TCOTrainingMixin:
    def init_tco(self, config=None): ...
    def apply_tco_weight(self, loss, gen_audio, ref_audio): ...
```

### 4.3 使用方法
```python
from f5_tts.restyle import TCOLoss, TCOConfig

# TCOLoss作成
tco_loss = TCOLoss(config=TCOConfig())

# 訓練ループ内
weighted_loss, metrics = tco_loss(
    base_loss,
    generated_audio=gen_audio,
    reference_audio=ref_audio,
)
```

---

## Phase 5: 推論インターフェース ✅ 完了

### 5.1 API拡張 (`api.py`) ✅
```python
class F5TTS:
    def load_style_loras(self, lora_paths: dict[str, str]):
        """Style LoRAを読み込む"""
        ...

    def infer(
        self,
        ref_file, ref_text, gen_text,
        # DCFG
        use_dcfg=False, lambda_t=2.0, lambda_a=0.5,
        # Style LoRA
        style_weights=None,  # {"pitch_high": 1.0, "angry": 0.5}
        use_olora=True,
    ):
        ...
```

### 5.2 推論ユーティリティ拡張 (`utils_infer.py`) ✅
- `infer_process()` にReStyleパラメータ追加
- `infer_batch_process()` にReStyleパラメータ追加
- Style LoRAコンテキストマネージャー統合

### 5.3 Gradio UI拡張 (`infer_gradio.py`) ✅
- DCFG設定パネル（日本語）
- スタイル制御スライダー（ピッチ、エネルギー）
- 感情選択ドロップダウン
- OLoRA融合トグル

### 5.4 学習ガイド (`docs/TRAINING_GUIDE.md`) ✅
- データセット準備手順
- 自動ラベリングスクリプト
- 学習コマンド例
- トラブルシューティング

---

## 実装順序

```
Phase 1: DCFG ✅ 完了
├── ✅ restyle/dcfg.py 作成
├── ✅ dit.py 修正（3パス推論）
├── ✅ cfm.py 修正（DCFGパラメータ）
└── ✅ テスト作成・検証

Phase 2: Style LoRA ✅ 完了
├── ✅ peft依存関係追加
├── ✅ restyle/style_lora.py 作成
├── ✅ train/train_style_lora.py 作成
├── ✅ configs/ReStyleTTS_Base.yaml 作成
└── ✅ テスト作成・検証

Phase 3: OLoRA Fusion ✅ 完了
├── ✅ olora_fusion.py 作成
├── ✅ StyleLoRAManagerへの統合
└── ✅ テスト作成・検証 (30テスト)

Phase 4: TCO ✅ 完了
├── ✅ speaker_encoder.py 作成
├── ✅ tco.py 作成
└── ✅ テスト作成・検証 (31テスト)

Phase 5: 推論インターフェース ✅ 完了
├── ✅ api.py 拡張
├── ✅ utils_infer.py 拡張
├── ✅ infer_gradio.py 拡張（日本語UI）
├── ✅ docs/TRAINING_GUIDE.md 作成
└── ✅ テスト（98テストパス）
```

---

## テスト状況

| テストファイル | テスト数 | 状態 |
|---------------|---------|------|
| `tests/test_dcfg.py` | 16 (14 passed, 2 skipped) | ✅ |
| `tests/test_style_lora.py` | 21 (21 passed) | ✅ |
| `tests/test_olora_fusion.py` | 30 (30 passed) | ✅ |
| `tests/test_tco.py` | 31 (30 passed, 1 skipped) | ✅ |
| **合計** | **98 (95 passed, 3 skipped)** | ✅ |

---

## 検証方法

### ユニットテスト
```bash
uv run pytest tests/ -v
```

### 手動テスト（推論）
```bash
# DCFG
uv run python -c "
from f5_tts.model.cfm import CFM
# use_dcfg=True で DCFG モード
"

# Style LoRA
uv run python -c "
from f5_tts.restyle import StyleLoRAManager
# マネージャーでLoRAを管理
"
```

---

## 更新履歴

| 日付 | 内容 |
|------|------|
| 2026-01-10 | Phase 4 (TCO) 完了 |
| 2026-01-09 | Phase 3 (OLoRA Fusion) 完了 |
| 2026-01-09 | Phase 2 (Style LoRA) 完了 |
| 2026-01-09 | Phase 1 (DCFG) 完了 |
| 2026-01-09 | 初版作成 |
