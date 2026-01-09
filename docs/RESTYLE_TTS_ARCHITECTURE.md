# ReStyle-TTS アーキテクチャ詳細

> **実装状況**: Phase 1-4 ✅ 完了 (DCFG, Style LoRA, OLoRA Fusion, TCO)
>
> 詳細は [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) を参照

## 1. 論文概要

**ReStyle-TTS: Relative and Continuous Style Control for Zero-Shot Speech Synthesis**
- arXiv:2601.03632v1 (2026年1月7日)
- 著者: Haitao Li et al. (Zhejiang University, Shanghai Jiao Tong University)

### 問題設定
ゼロショットTTSでは参照音声のスタイル（韻律・感情）が生成音声に強く影響し、柔軟なスタイル制御が困難。

### 解決策
3つのコンポーネントによる段階的アプローチ:
1. **DCFG**: 参照音声への依存を減らす
2. **Style LoRA + OLoRA**: 明示的で連続的なスタイル制御
3. **TCO**: 音色劣化を補償

---

## 2. F5-TTS ベースモデル構造

### 2.1 全体アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                        CFM クラス                           │
│  (src/f5_tts/model/cfm.py)                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  MelSpec    │    │ Transformer │    │  ODE Solver │     │
│  │  変換       │ -> │   (DiT)     │ -> │ (odeint)    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 DiT (Diffusion Transformer) 構造

**ファイル**: `src/f5_tts/model/backbones/dit.py`

```
DiT
├── text_embed: TextEmbedding
│   ├── nn.Embedding(vocab_size, text_dim)
│   └── ConvNeXtV2Block × conv_layers (オプション)
│
├── input_embed: InputEmbedding
│   ├── proj: nn.Linear(mel_dim*2 + text_dim, dim)
│   └── conv_pos_embed: ConvPositionEmbedding
│
├── rotary_embed: RotaryEmbedding
│
├── time_embed: TimestepEmbedding
│   └── time_mlp: nn.Sequential(Linear, SiLU, Linear)
│
├── transformer_blocks: nn.ModuleList
│   └── DiTBlock × depth (22 layers)
│       ├── attn_norm: AdaLayerNorm
│       │   └── linear: nn.Linear(dim, dim*6)
│       ├── attn: Attention
│       │   ├── to_q: nn.Linear(dim, inner_dim)
│       │   ├── to_k: nn.Linear(dim, inner_dim)
│       │   ├── to_v: nn.Linear(dim, inner_dim)
│       │   └── to_out: nn.Linear(inner_dim, dim)
│       └── ff: FeedForward
│           ├── ff[0]: nn.Linear(dim, inner_dim)
│           └── ff[2]: nn.Linear(inner_dim, dim)
│
├── norm_out: RMSNorm
└── proj_out: nn.Linear(dim, mel_dim)
```

### 2.3 モデルパラメータ (F5TTS_v1_Base)

| パラメータ | 値 |
|-----------|-----|
| dim | 1024 |
| depth | 22 |
| heads | 16 |
| dim_head | 64 |
| ff_mult | 2 |
| text_dim | 512 |
| mel_dim (n_mel_channels) | 100 |
| conv_layers | 4 |

### 2.4 線形層の総数

1ブロックあたり:
- AdaLayerNorm.linear: 1
- Attention (to_q, to_k, to_v, to_out): 4
- FeedForward: 2
- **合計**: 7層/ブロック × 22ブロック = **154層**

その他:
- InputEmbedding.proj: 1
- TimestepEmbedding: 2
- proj_out: 1
- TextEmbedding内: 複数

---

## 3. 現在のCFG実装

### 3.1 訓練時のドロップアウト

**ファイル**: `cfm.py:286-296`

```python
# ドロップ確率
audio_drop_prob = 0.3  # 参照音声をドロップ
cond_drop_prob = 0.2   # 両方（音声+テキスト）をドロップ

# 訓練時の適用
drop_audio_cond = random() < self.audio_drop_prob
if random() < self.cond_drop_prob:
    drop_audio_cond = True
    drop_text = True
else:
    drop_text = False
```

### 3.2 推論時のCFG

**ファイル**: `cfm.py:160-191`

```python
def fn(t, x):
    if cfg_strength < 1e-5:
        # CFGなし: 条件付き予測のみ
        pred = self.transformer(x, cond, text, t,
                                drop_audio_cond=False, drop_text=False)
        return pred

    # CFGあり: バッチ連結で効率化
    pred_cfg = self.transformer(x, cond, text, t, cfg_infer=True)
    pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
    return pred + (pred - null_pred) * cfg_strength
```

### 3.3 DiTでのcfg_infer処理

**ファイル**: `dit.py:296-310`

```python
if cfg_infer:
    # 条件付きと無条件を連結
    x_cond = self.get_input_embed(x, cond, text,
                                   drop_audio_cond=False, drop_text=False)
    x_uncond = self.get_input_embed(x, cond, text,
                                     drop_audio_cond=True, drop_text=True)
    x = torch.cat((x_cond, x_uncond), dim=0)
    t = torch.cat((t, t), dim=0)
```

---

## 4. ReStyle-TTS コンポーネント詳細

### 4.1 DCFG (Decoupled Classifier-Free Guidance) ✅ 実装済み

**実装ファイル**:
- `src/f5_tts/restyle/dcfg.py` - DCFGConfig, dcfg_combine関数
- `src/f5_tts/model/backbones/dit.py` - dcfg_inferパラメータ追加
- `src/f5_tts/model/cfm.py` - sample()にuse_dcfg, lambda_t, lambda_a追加

#### 数式

**従来のCFG**:
```
f̂_CFG = f_{a,t} + λ_cfg × (f_{a,t} - f_{∅,∅})
```

**DCFG**:
```
f̂_DCFG = f_{∅,t} + λ_t × (f_{∅,t} - f_{∅,∅}) + λ_a × (f_{a,t} - f_{∅,t})
```

- `f_{∅,∅}`: 無条件（音声もテキストもドロップ）
- `f_{∅,t}`: テキストのみ条件（音声ドロップ）
- `f_{a,t}`: フル条件（両方あり）
- `λ_t`: テキストガイダンス強度 (デフォルト: 2.0)
- `λ_a`: 参照音声ガイダンス強度 (デフォルト: 0.5)

#### 等価性

`λ_t = λ_cfg`, `λ_a = 1 + λ_cfg` のとき、DCFGは従来CFGと等価。

#### 訓練時の変更

```python
# 3段階のドロップアウト
drop_text_prob = 0.15      # テキストのみドロップ
drop_audio_prob = 0.15     # 音声のみドロップ
drop_both_prob = 0.2       # 両方ドロップ

# 各条件の生成
# 1. f_{∅,∅}: drop_audio=True, drop_text=True
# 2. f_{∅,t}: drop_audio=True, drop_text=False
# 3. f_{a,t}: drop_audio=False, drop_text=False
```

### 4.2 Style LoRA ✅ 実装済み

**実装ファイル**:
- `src/f5_tts/restyle/style_lora.py` - StyleLoRAManager, StyleLoRAConfig
- `src/f5_tts/train/train_style_lora.py` - LoRA訓練スクリプト
- `src/f5_tts/configs/ReStyleTTS_Base.yaml` - 設定ファイル

#### LoRA設定

```python
lora_config = {
    "r": 32,           # ランク
    "lora_alpha": 64,  # スケーリング係数
    "target_modules": [
        "to_q", "to_k", "to_v", "to_out",  # Attention
        "ff.0", "ff.2",                     # FeedForward
    ],
    "lora_dropout": 0.0,
}
```

#### 学習するスタイル属性

| カテゴリ | 属性 |
|---------|------|
| Pitch | high, low |
| Energy | high, low |
| Emotion | angry, happy, sad, fear, disgusted, surprised |

#### 訓練データ

- VccmDataset (LibriTTS + 感情データセット)
- 訓練時間: 各属性250時間

### 4.3 OLoRA Fusion (Orthogonal LoRA Fusion) ✅ 実装済み

**実装ファイル**:
- `src/f5_tts/restyle/olora_fusion.py` - OLoRAFusion, fuse_lora_weights
- `src/f5_tts/restyle/style_lora.py` - StyleLoRAManagerへの統合 (use_olora)

#### アルゴリズム

```python
def olora_fusion(lora_weights: List[Tensor], alphas: List[float]) -> Tensor:
    """
    複数のLoRAを直交射影して融合

    Args:
        lora_weights: [ΔW_1, ΔW_2, ..., ΔW_N] 各LoRAの重み
        alphas: [α_1, α_2, ..., α_N] 各LoRAの強度

    Returns:
        ΔW_fuse: 融合されたLoRA重み
    """
    N = len(lora_weights)
    D = lora_weights[0].numel()

    # ベクトル化
    V = torch.stack([w.flatten() for w in lora_weights])  # [N, D]

    # 各LoRAを他のLoRAの部分空間に直交射影
    orthogonalized = []
    for i in range(N):
        # V_{-i}: i番目以外のLoRA
        V_minus_i = torch.cat([V[:i], V[i+1:]], dim=0)  # [N-1, D]

        # 射影行列: P_{-i} = V_{-i}^T @ pinv(V_{-i}^T)
        P_minus_i = V_minus_i.T @ torch.linalg.pinv(V_minus_i.T)

        # 直交化: v̂_i = (I - P_{-i}) @ v_i
        v_i = V[i]
        v_orth = v_i - P_minus_i @ v_i
        orthogonalized.append(v_orth)

    # 重み付き合成
    fused = sum(alpha * v for alpha, v in zip(alphas, orthogonalized))
    return fused.reshape(lora_weights[0].shape)
```

#### 特徴

- **順序非依存**: 全てのLoRAを同時に直交化
- **疎性保証**: N << D のため効果的に干渉を除去
- **訓練不要**: 推論時にのみ適用

### 4.4 TCO (Timbre Consistency Optimization) ✅ 実装済み

**実装ファイル**:
- `src/f5_tts/restyle/speaker_encoder.py` - WavLM話者エンコーダー
- `src/f5_tts/restyle/tco.py` - TCO損失実装

#### 目的

DCFGで参照音声への依存を減らすと音色の一貫性が低下する問題を補償。

#### 数式

```python
# 標準のFlow Matching損失
L_FM = E[(f_θ(x) - y)²]

# 話者類似度報酬
r_t = speaker_similarity(generated_audio, reference_audio)

# EMAベースライン
b_t = μ * b_{t-1} + (1 - μ) * r_t

# アドバンテージ
A_t = r_t - b_t

# 報酬重み付け
w_t = 1 + λ * tanh(β * A_t)

# 最終損失
L_total = w_t * L_FM
```

#### ハイパーパラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| λ | 0.2 | 報酬強度 |
| β | 5.0 | アドバンテージ感度 |
| μ | 0.9 | EMAモメンタム |

#### 話者類似度計算

```python
# WavLM base-plus-sv を使用
from transformers import WavLMForXVector

speaker_model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")

def speaker_similarity(audio1, audio2):
    emb1 = speaker_model(audio1).embeddings
    emb2 = speaker_model(audio2).embeddings
    return F.cosine_similarity(emb1, emb2)
```

---

## 5. 推論パイプライン

### 5.1 標準F5-TTS推論

```
Reference Audio → MelSpec → CFM.sample() → Vocoder → Output Audio
                              ↑
                           Text Input
```

### 5.2 ReStyle-TTS推論

```
Reference Audio → MelSpec ─┐
                           │
Text Input ────────────────┼─→ DiT + Style LoRAs → DCFG → Vocoder → Output
                           │        ↑
Style Controls ────────────┘   OLoRA Fusion
(pitch, energy, emotion)            ↑
                              [LoRA_pitch, LoRA_energy, LoRA_emotion]
```

### 5.3 推論パラメータ

```python
# DCFG パラメータ
lambda_t = 2.0      # テキストガイダンス
lambda_a = 0.5      # 参照音声ガイダンス

# Style LoRA パラメータ
alpha_pitch = 0.0   # [-2.0, 2.0] 負で逆効果
alpha_energy = 0.0  # [-2.0, 2.0]
alpha_emotion = 0.0 # [0.0, 4.0] 感情強度

# ODE パラメータ
nfe_step = 32       # Number of Function Evaluations
ode_method = "euler"
```

---

## 6. 評価指標

### 6.1 客観評価

| 指標 | 計算方法 | ツール |
|------|----------|--------|
| WER | 文字誤り率 | Whisper-large-v3 |
| Spk-sv | 話者類似度 | WavLM base-plus-sv |
| Pitch | 基本周波数 | Parselmouth |
| Energy | STFT振幅のL2ノルム | - |
| Emotion | 感情分類 | Emotion2Vec |

### 6.2 主観評価

- **MOS-SA**: Mean Opinion Score - Style Accuracy (1-5スケール)

---

## 7. 参考文献

1. F5-TTS: Chen et al. (2024) - F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching
2. ReStyle-TTS: Li et al. (2026) - ReStyle-TTS: Relative and Continuous Style Control for Zero-Shot Speech Synthesis
3. LoRA: Hu et al. (2021) - LoRA: Low-Rank Adaptation of Large Language Models
4. Flow Matching: Lipman et al. (2022) - Flow Matching for Generative Modeling
