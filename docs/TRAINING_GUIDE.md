# ReStyle-TTS Style LoRA 学習ガイド

本ドキュメントでは、ReStyle-TTS の Style LoRA アダプターを学習する手順を説明します。

## 目次

1. [概要](#1-概要)
2. [必要な環境](#2-必要な環境)
3. [データセットの準備](#3-データセットの準備)
4. [学習の実行](#4-学習の実行)
5. [学習済みモデルの使用](#5-学習済みモデルの使用)
6. [トラブルシューティング](#6-トラブルシューティング)

---

## 1. 概要

### Style LoRA とは

Style LoRA は、ベースの F5-TTS モデルに対して低ランク適応（LoRA）を行い、特定のスタイル属性を制御するためのアダプターです。

### 学習可能なスタイル属性

| カテゴリ | 属性 | 説明 |
|---------|------|------|
| **ピッチ** | `pitch_high` | 高いピッチの音声 |
| | `pitch_low` | 低いピッチの音声 |
| **エネルギー** | `energy_high` | 大きな声・強いエネルギー |
| | `energy_low` | 小さな声・弱いエネルギー |
| **感情** | `angry` | 怒り |
| | `happy` | 喜び |
| | `sad` | 悲しみ |
| | `fear` | 恐怖 |
| | `disgusted` | 嫌悪 |
| | `surprised` | 驚き |

### 学習の流れ

```
1. データセット準備
   ├── 音声ファイル収集
   ├── スタイルラベリング
   └── Arrow形式への変換

2. ベースモデル取得
   └── F5TTS_v1_Base をダウンロード

3. Style LoRA 学習
   ├── ベースモデル凍結
   ├── LoRA層のみ学習
   └── チェックポイント保存

4. 推論での使用
   └── LoRAを読み込んでスタイル制御
```

---

## 2. 必要な環境

### ハードウェア要件

| 項目 | 最小 | 推奨 |
|------|------|------|
| GPU VRAM | 8GB | 16GB以上 |
| RAM | 16GB | 32GB以上 |
| ストレージ | 50GB | 100GB以上 |

### ソフトウェア要件

```bash
# Python 3.12.7
uv run python --version

# CUDA 12.x
nvcc --version

# 依存関係インストール
uv sync

# WandB ログイン（必須）
wandb login
```

**注意**: 学習時のログ記録には WandB が必須です。アカウントを作成し、ログインしてください。

### ベースモデルのダウンロード

```bash
# HuggingFace CLI でダウンロード
huggingface-cli download SWivid/F5-TTS \
    F5TTS_v1_Base/model_1200000.safetensors \
    --local-dir checkpoints/

# または Python から
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="SWivid/F5-TTS",
    filename="F5TTS_v1_Base/model_1200000.safetensors",
    local_dir="checkpoints/"
)
```

---

## 3. データセットの準備

### 3.1 論文再現用データセット（必須）

ReStyle-TTS論文（arXiv:2601.03632）を再現するには、以下の **英語データセット** を使用します。
論文では「VccmDataset」という統合データセットを構築しています。

| データセット | サイズ | 用途 | ダウンロード |
|-------------|--------|------|-------------|
| **LibriTTS** | ~585時間 | ピッチ/エネルギー | [OpenSLR](https://openslr.org/60/) |
| **VCTK** | 110話者 | ピッチ/エネルギー | [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/2950) |
| **TESS** | 2,800ファイル | 感情（7カテゴリ） | [Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) |

**重要**: 論文では **英語のみ** で学習・評価されています。

### 3.2 TESSデータセットの構造

TESSには以下の7つの感情カテゴリが含まれます：

| 感情 | スタイル属性名 | ファイル数 |
|------|--------------|-----------|
| Angry | `angry` | 400 |
| Disgust | `disgusted` | 400 |
| Fear | `fear` | 400 |
| Happy | `happy` | 400 |
| Sad | `sad` | 400 |
| Pleasant Surprise | `surprised` | 400 |
| Neutral | `neutral` | 400 |

TESSのディレクトリ構造：
```
TESS/
├── OAF_angry/
├── OAF_disgust/
├── OAF_fear/
├── OAF_happy/
├── OAF_neutral/
├── OAF_pleasant_surprised/
├── OAF_sad/
├── YAF_angry/
├── YAF_disgust/
...
```

### 3.3 代替データセット（実験用）

論文再現以外の実験や、他言語で試す場合：

#### 英語（代替）

| データセット | サイズ | 用途 | ダウンロード |
|-------------|--------|------|-------------|
| ESD | 29時間 | 感情 | [GitHub](https://github.com/HLTSingapore/Emotional-Speech-Data) |
| RAVDESS | 7GB | 感情 | [Zenodo](https://zenodo.org/record/1188976) |

#### 日本語（実験用）

| データセット | サイズ | 用途 | ダウンロード |
|-------------|--------|------|-------------|
| JVS Corpus | 30時間 | 全般 | [公式サイト](https://sites.google.com/site/shinaborulab/research-topics/jvs_corpus) |
| JSUT | 10時間 | 全般 | [公式サイト](https://sites.google.com/site/shinaborulab/research-topics/jsut) |

### 3.4 データセット構造（共通）

学習データセットは以下の構造で準備します：

```
datasets/
├── pitch_high/
│   ├── wavs/
│   │   ├── audio_001.wav
│   │   ├── audio_002.wav
│   │   └── ...
│   └── metadata.csv
├── pitch_low/
│   └── ...
├── angry/
│   └── ...
└── happy/
    └── ...
```

#### metadata.csv の形式

```csv
audio_file|text
audio_001.wav|これはサンプルテキストです。
audio_002.wav|音声合成のテストです。
```

**注意**:
- 音声ファイルは 24kHz、モノラル推奨
- テキストはひらがな/漢字混じりまたはピンイン

### 3.5 自動ラベリング（ピッチ/エネルギー）

ピッチとエネルギーは音声から自動的にラベリングできます。

```python
# scripts/label_pitch_energy.py

import parselmouth
import librosa
import numpy as np
from pathlib import Path
import json

def compute_pitch_stats(audio_path):
    """音声ファイルのピッチ統計を計算"""
    snd = parselmouth.Sound(str(audio_path))
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values > 0]  # 無声区間を除外

    if len(pitch_values) == 0:
        return None

    return {
        'mean': float(np.mean(pitch_values)),
        'std': float(np.std(pitch_values)),
        'max': float(np.max(pitch_values)),
        'min': float(np.min(pitch_values)),
    }

def compute_energy(audio_path):
    """音声ファイルのRMSエネルギーを計算"""
    y, sr = librosa.load(str(audio_path), sr=24000)
    rms = np.sqrt(np.mean(y**2))
    return float(rms)

def label_dataset(input_dir, output_dir):
    """データセットをラベリング"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 全ファイルの統計を収集
    stats = []
    for wav_file in input_path.glob("wavs/*.wav"):
        pitch = compute_pitch_stats(wav_file)
        energy = compute_energy(wav_file)
        if pitch:
            stats.append({
                'file': wav_file.name,
                'pitch_mean': pitch['mean'],
                'energy': energy,
            })

    # 上位/下位20%でラベリング
    pitch_values = [s['pitch_mean'] for s in stats]
    energy_values = [s['energy'] for s in stats]

    pitch_high_threshold = np.percentile(pitch_values, 80)
    pitch_low_threshold = np.percentile(pitch_values, 20)
    energy_high_threshold = np.percentile(energy_values, 80)
    energy_low_threshold = np.percentile(energy_values, 20)

    # 結果を保存
    labels = {
        'pitch_high': [s['file'] for s in stats if s['pitch_mean'] >= pitch_high_threshold],
        'pitch_low': [s['file'] for s in stats if s['pitch_mean'] <= pitch_low_threshold],
        'energy_high': [s['file'] for s in stats if s['energy'] >= energy_high_threshold],
        'energy_low': [s['file'] for s in stats if s['energy'] <= energy_low_threshold],
    }

    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / 'labels.json', 'w') as f:
        json.dump(labels, f, indent=2)

    print(f"Labeled {len(stats)} files")
    for label, files in labels.items():
        print(f"  {label}: {len(files)} files")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: uv run python label_pitch_energy.py <input_dir> <output_dir>")
        sys.exit(1)
    label_dataset(sys.argv[1], sys.argv[2])
```

### 3.6 Arrow形式への変換

F5-TTS の学習にはArrow形式のデータセットが必要です。

```bash
# CSV + WAV → Arrow形式
uv run python -m f5_tts.train.datasets.prepare_csv_wavs \
    datasets/pitch_high \
    datasets/pitch_high_prepared \
    --workers 16
```

出力構造：
```
datasets/pitch_high_prepared/
├── raw.arrow      # HuggingFace Arrow形式
├── duration.json  # 各サンプルの長さ
└── vocab.txt      # 語彙ファイル
```

---

## 4. 学習の実行

### 4.1 基本コマンド

```bash
uv run python -m f5_tts.train.train_style_lora \
    --config-name ReStyleTTS_Base \
    style_attribute=pitch_high \
    pretrained_checkpoint=checkpoints/F5TTS_v1_Base/model_1200000.safetensors \
    datasets.name=pitch_high_prepared
```

### 4.2 主要パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `style_attribute` | (必須) | 学習するスタイル属性 |
| `pretrained_checkpoint` | (必須) | ベースモデルのパス |
| `datasets.name` | (必須) | データセット名 |
| `optim.epochs` | 11 | エポック数 |
| `optim.lora_learning_rate` | 1e-5 | 学習率 |
| `lora.rank` | 32 | LoRAランク |
| `lora.alpha` | 64 | LoRAスケーリング係数 |

### 4.3 カスタム設定例

```bash
# 高速学習（小さいLoRAランク）
uv run python -m f5_tts.train.train_style_lora \
    --config-name ReStyleTTS_Base \
    style_attribute=angry \
    pretrained_checkpoint=checkpoints/model_1200000.safetensors \
    datasets.name=angry_prepared \
    lora.rank=16 \
    optim.epochs=5 \
    optim.lora_learning_rate=2e-5

# 高品質学習（大きいLoRAランク）
uv run python -m f5_tts.train.train_style_lora \
    --config-name ReStyleTTS_Base \
    style_attribute=happy \
    pretrained_checkpoint=checkpoints/model_1200000.safetensors \
    datasets.name=happy_prepared \
    lora.rank=64 \
    optim.epochs=20 \
    optim.lora_learning_rate=5e-6
```

### 4.4 マルチGPU学習

```bash
# Accelerateを使用
accelerate launch --multi_gpu --num_processes 2 \
    -m f5_tts.train.train_style_lora \
    --config-name ReStyleTTS_Base \
    style_attribute=pitch_high \
    pretrained_checkpoint=checkpoints/model_1200000.safetensors
```

### 4.5 全スタイル一括学習スクリプト

```bash
#!/bin/bash
# scripts/train_all_styles.sh

STYLES=(
    "pitch_high"
    "pitch_low"
    "energy_high"
    "energy_low"
    "angry"
    "happy"
    "sad"
)

CHECKPOINT="checkpoints/F5TTS_v1_Base/model_1200000.safetensors"

for style in "${STYLES[@]}"; do
    echo "Training $style..."
    uv run python -m f5_tts.train.train_style_lora \
        --config-name ReStyleTTS_Base \
        style_attribute=$style \
        pretrained_checkpoint=$CHECKPOINT \
        datasets.name=${style}_prepared

    echo "$style completed!"
done
```

### 4.6 チェックポイント

学習済みモデルは以下に保存されます：

```
ckpts/ReStyleTTS_Base/
├── lora_pitch_high_10000.safetensors
├── lora_pitch_high_20000.safetensors
├── lora_pitch_high_last.safetensors
└── ...
```

### 4.7 論文再現設定

ReStyle-TTS論文では、各スタイル属性につき **250時間** の学習を実施しています。
これはエポック数ではなく、総学習時間で管理されています。

```bash
# 論文再現用（250時間学習）
uv run python -m f5_tts.train.train_style_lora \
    --config-name ReStyleTTS_Base \
    style_attribute=pitch_high \
    pretrained_checkpoint=checkpoints/F5TTS_v1_Base/model_1200000.safetensors \
    datasets.name=pitch_high_prepared \
    optim.max_hours=250
```

**論文の学習設定:**

| パラメータ | 値 |
|-----------|-----|
| 学習時間 | 250時間/スタイル属性 |
| LoRAランク | 32 |
| LoRAアルファ | 64 |
| サンプリングレート | 24kHz |
| 言語 | 英語のみ |

---

## 5. 学習済みモデルの使用

### 5.1 Python API

```python
from f5_tts.api import F5TTS
from f5_tts.restyle import StyleLoRAManager

# モデル読み込み
tts = F5TTS()

# Style LoRA読み込み
manager = StyleLoRAManager(tts.ema_model.transformer)
manager.load_lora("pitch_high", "ckpts/ReStyleTTS_Base/lora_pitch_high_last.safetensors")
manager.load_lora("angry", "ckpts/ReStyleTTS_Base/lora_angry_last.safetensors")

# スタイル適用して推論
with manager.apply_styles({"pitch_high": 1.0, "angry": 0.5}, use_olora=True):
    audio, sr = tts.infer(
        ref_file="reference.wav",
        ref_text="参照テキスト",
        gen_text="生成したいテキスト",
        use_dcfg=True,
        lambda_t=2.0,
        lambda_a=0.5,
    )
```

### 5.2 Gradio UI

Gradio UIを起動すると、ReStyle設定セクションでスタイルを制御できます：

```bash
uv run python src/f5_tts/infer/infer_gradio.py
```

---

## 6. トラブルシューティング

### よくある問題

#### CUDA out of memory

```
解決策:
1. batch_size_per_gpu を減らす
   datasets.batch_size_per_gpu=19200
2. gradient_accumulation_steps を増やす
   optim.grad_accumulation_steps=2
3. lora.rank を減らす
   lora.rank=16
```

#### データセットが見つからない

```
解決策:
1. datasets.name が正しいか確認
2. Arrow形式への変換が完了しているか確認
3. vocab.txt が存在するか確認
```

#### 学習が収束しない

```
解決策:
1. 学習率を下げる
   optim.lora_learning_rate=5e-6
2. エポック数を増やす
   optim.epochs=20
3. データセットの品質を確認
```

### ログの確認

```bash
# WandB ダッシュボード
# 学習開始後、https://wandb.ai/ でログを確認できます

# TensorBoard（オプション）
tensorboard --logdir ckpts/ReStyleTTS_Base/
```

---

## 参考リンク

- [F5-TTS 公式リポジトリ](https://github.com/SWivid/F5-TTS)
- [ReStyle-TTS 論文](https://arxiv.org/abs/2601.03632)
- [LoRA 論文](https://arxiv.org/abs/2106.09685)
- [PEFT ライブラリ](https://github.com/huggingface/peft)
