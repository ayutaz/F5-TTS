# ReStyle-TTS 実装ロードマップ

F5-TTSにReStyle-TTS（arXiv:2601.03632）の機能を追加実装するためのロードマップ。

## 概要

ReStyle-TTSは、ゼロショット音声合成における連続的かつ相対的なスタイル制御を実現する手法。以下の4つの主要技術で構成される：

1. **DCFG** - テキストと参照音声のガイダンスを分離
2. **Style LoRA** - スタイル属性ごとのアダプター
3. **OLoRA Fusion** - 複数LoRAの直交融合
4. **TCO** - 音色一貫性の最適化

---

## 実装状況

| Phase | 機能 | 状態 | ブランチ |
|-------|------|------|---------|
| 1 | DCFG | ✅ 完了 | `feature/restyle-dcfg` |
| 2 | Style LoRA | ✅ 完了 | `feature/restyle-dcfg` |
| 3 | OLoRA Fusion | ✅ 完了 | `feature/restyle-dcfg` |
| 4 | TCO | ✅ 完了 | `feature/restyle-dcfg` |
| 5 | 推論インターフェース | ✅ 完了 | `feature/restyle-dcfg` |

---

## Phase 1: DCFG (Decoupled Classifier-Free Guidance) ✅

### 目的
テキストと参照音声のガイダンスを分離し、参照スタイルへの依存度を制御可能にする。

### 実装内容
- [x] `src/f5_tts/restyle/dcfg.py` - DCFG設定とユーティリティ
- [x] `src/f5_tts/model/backbones/dit.py` - 3パス推論（dcfg_infer）
- [x] `src/f5_tts/model/cfm.py` - sample()にDCFGパラメータ追加
- [x] `tests/test_dcfg.py` - ユニットテスト（14テストパス）

### 使用方法
```python
from f5_tts.model.cfm import CFM

# DCFG有効
output, _ = model.sample(
    cond, text, duration,
    use_dcfg=True,
    lambda_t=2.0,  # テキストガイダンス強度
    lambda_a=0.5,  # 参照音声ガイダンス強度
)
```

### パラメータ
| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `use_dcfg` | `False` | DCFGを有効にするか |
| `lambda_t` | `2.0` | テキストガイダンス強度 |
| `lambda_a` | `0.5` | 参照音声ガイダンス強度（0で影響なし） |

---

## Phase 2: Style LoRA ✅

### 目的
各スタイル属性（ピッチ、エネルギー、感情）に特化したLoRAアダプターを学習し、連続的なスタイル制御を可能にする。

### 実装内容
- [x] `peft>=0.7.0` 依存関係追加
- [x] `src/f5_tts/restyle/style_lora.py` - StyleLoRAManager実装
- [x] `src/f5_tts/train/train_style_lora.py` - LoRA訓練スクリプト
- [x] `src/f5_tts/configs/ReStyleTTS_Base.yaml` - 設定ファイル
- [x] `tests/test_style_lora.py` - テスト（21テストパス）

### 使用方法
```python
from f5_tts.restyle import StyleLoRAManager, StyleLoRAConfig

# マネージャー初期化
manager = StyleLoRAManager(model.transformer)

# LoRAを読み込み
manager.load_lora("pitch_high", "path/to/pitch_high.safetensors")
manager.load_lora("angry", "path/to/angry.safetensors")

# スタイルを適用して推論
with manager.apply_styles({"pitch_high": 1.0, "angry": 0.5}):
    output = model.sample(...)
```

### 訓練方法
```bash
python -m f5_tts.train.train_style_lora --config-name ReStyleTTS_Base \
    style_attribute=pitch_high \
    pretrained_checkpoint=path/to/base_model.pt
```

### スタイル属性
| カテゴリ | 属性 |
|---------|------|
| ピッチ | `pitch_high`, `pitch_low` |
| エネルギー | `energy_high`, `energy_low` |
| 感情 | `angry`, `happy`, `sad`, `fear`, `disgusted`, `surprised` |

### LoRA設定
```yaml
lora:
  rank: 32
  alpha: 64
  target_modules:
    - to_q
    - to_k
    - to_v
    - to_out.0
    - ff.ff.0
    - ff.ff.2
  dropout: 0.0
```

---

## Phase 3: OLoRA Fusion ✅

### 目的
複数のStyle LoRAを干渉なく同時適用するための直交融合メカニズム。

### 実装内容
- [x] `src/f5_tts/restyle/olora_fusion.py` - 直交射影実装
- [x] StyleLoRAManagerへの統合（`use_olora`パラメータ）
- [x] `tests/test_olora_fusion.py` - テスト（30テストパス）

### 数式
```
v̂_i = (I - P_{-i}) @ v_i
P_{-i} = V_{-i}^T @ pinv(V_{-i}^T)
ΔW_fuse = Σ α_i * ΔŴ_i
```

### 使用方法
```python
from f5_tts.restyle import StyleLoRAManager, OLoRAConfig

# マネージャー初期化（OLoRA設定付き）
manager = StyleLoRAManager(model.transformer, olora_config=OLoRAConfig())

# LoRAを読み込み
manager.load_lora("pitch_high", "path/to/pitch_high.safetensors")
manager.load_lora("angry", "path/to/angry.safetensors")

# OLoRA有効で複数スタイルを適用
with manager.apply_styles({"pitch_high": 1.0, "angry": 0.5}, use_olora=True):
    output = model.sample(...)

# OLoRA無効（通常の重み付き合成）
with manager.apply_styles({"pitch_high": 1.0}, use_olora=False):
    output = model.sample(...)
```

### OLoRAFusion クラス（低レベルAPI）
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

## Phase 4: TCO (Timbre Consistency Optimization) ✅

### 目的
DCFGで参照依存を減らした際の音色劣化を補償する。

### 実装内容
- [x] `src/f5_tts/restyle/speaker_encoder.py` - WavLM話者エンコーダー
- [x] `src/f5_tts/restyle/tco.py` - アドバンテージ重み付き損失
- [x] `tests/test_tco.py` - テスト（31テスト、30 passed, 1 skipped）

### 数式
```
w_t = 1 + λ * tanh(β * A_t)
L_total = w_t * L_FM
A_t = r_t - b_t  (アドバンテージ = 報酬 - ベースライン)
b_t = μ * b_{t-1} + (1 - μ) * r_t  (EMAベースライン)
```

### ハイパーパラメータ
| パラメータ | 値 | 説明 |
|-----------|-----|------|
| λ | 0.2 | 報酬強度 |
| β | 5.0 | アドバンテージ感度 |
| μ | 0.9 | EMAモメンタム |

### 使用方法
```python
from f5_tts.restyle import TCOLoss, TCOConfig, SpeakerEncoder

# TCO設定
config = TCOConfig(
    lambda_reward=0.2,
    beta=5.0,
    mu=0.9,
)

# TCOLoss作成
tco_loss = TCOLoss(config=config)

# 訓練ループ内
base_loss = compute_flow_matching_loss(...)
weighted_loss, metrics = tco_loss(
    base_loss,
    generated_audio=gen_audio,
    reference_audio=ref_audio,
)

# または事前計算報酬を使用
reward = compute_speaker_similarity(gen_audio, ref_audio)
weighted_loss, metrics = tco_loss(base_loss, reward=reward)
```

---

## Phase 5: 推論インターフェース ✅

### 目的
ReStyle-TTS機能をAPI、CLI、Gradio UIから利用可能にする。

### 実装内容

#### 5.1 API拡張 (`api.py`)
- [x] `use_dcfg`, `lambda_t`, `lambda_a` パラメータ追加
- [x] `style_weights` パラメータ追加（スタイル重み辞書）
- [x] `use_olora` パラメータ追加
- [x] `load_style_loras()` メソッド追加
- [x] `get_loaded_styles()` メソッド追加

#### 5.2 推論ユーティリティ拡張 (`utils_infer.py`)
- [x] `infer_process()` にReStyleパラメータ追加
- [x] `infer_batch_process()` にReStyleパラメータ追加
- [x] Style LoRAコンテキストマネージャー統合

#### 5.3 Gradio UI拡張 (`infer_gradio.py`)
- [x] DCFG設定パネル追加（日本語）
- [x] スタイル制御スライダー追加（ピッチ、エネルギー）
- [x] 感情選択ドロップダウン追加
- [x] OLoRA融合トグル追加

#### 5.4 学習ガイド
- [x] `docs/TRAINING_GUIDE.md` - 学習手順ドキュメント

### 使用方法

**Python API:**
```python
from f5_tts.api import F5TTS

tts = F5TTS()

# Style LoRAを読み込み（オプション）
tts.load_style_loras({
    "pitch_high": "path/to/pitch_high.safetensors",
    "angry": "path/to/angry.safetensors",
})

# DCFG + Style LoRAで推論
audio, sr, _ = tts.infer(
    ref_file="reference.wav",
    ref_text="参照テキスト",
    gen_text="生成したいテキスト",
    use_dcfg=True,
    lambda_t=2.0,
    lambda_a=0.5,
    style_weights={"pitch_high": 1.0, "angry": 0.5},
    use_olora=True,
)
```

**Gradio UI:**
```bash
python src/f5_tts/infer/infer_gradio.py
# ブラウザで「ReStyle設定」セクションを開く
```

---

## 依存関係

### 既存
- torch>=2.0.0
- torchaudio>=2.0.0
- transformers

### 追加予定
| パッケージ | Phase | 用途 |
|-----------|-------|------|
| peft>=0.7.0 | 2 | LoRAアダプター |

---

## 評価指標

| 指標 | ツール | 説明 |
|------|--------|------|
| WER | Whisper-large-v3 | 音声認識精度 |
| Spk-sv | WavLM base-plus-sv | 話者類似度 |
| Pitch | Parselmouth | ピッチ制御精度 |
| Energy | STFT L2 norm | エネルギー制御精度 |
| Emotion | Emotion2Vec | 感情認識精度 |

---

## 参考文献

- ReStyle-TTS: arXiv:2601.03632
- F5-TTS: Chen et al. 2024
- Flow Matching: Lipman et al. 2022
- LoRA: Hu et al. 2021

---

## 更新履歴

| 日付 | 内容 |
|------|------|
| 2026-01-10 | Phase 5 (推論インターフェース) 完了 |
| 2026-01-10 | Phase 4 (TCO) 完了 |
| 2026-01-09 | Phase 3 (OLoRA Fusion) 完了 |
| 2026-01-09 | Phase 2 (Style LoRA) 完了 |
| 2026-01-09 | Phase 1 (DCFG) 完了 |
| 2026-01-09 | ロードマップ作成 |
