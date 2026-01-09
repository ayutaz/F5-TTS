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

| カテゴリ | 属性 | 説明 | データソース |
|---------|------|------|-------------|
| **ピッチ** | `pitch_high` | 高いピッチの音声 | LibriTTS + VCTK |
| | `pitch_low` | 低いピッチの音声 | LibriTTS + VCTK |
| **エネルギー** | `energy_high` | 大きな声・強いエネルギー | LibriTTS + VCTK |
| | `energy_low` | 小さな声・弱いエネルギー | LibriTTS + VCTK |
| **感情** | `angry` | 怒り | TESS |
| | `happy` | 喜び | TESS |
| | `sad` | 悲しみ | TESS |
| | `fear` | 恐怖 | TESS |
| | `disgusted` | 嫌悪 | TESS |
| | `surprised` | 驚き | TESS |

---

## 2. 必要な環境

### ハードウェア要件

| 項目 | 最小 | 推奨 |
|------|------|------|
| GPU VRAM | 8GB | 16GB以上 |
| RAM | 16GB | 32GB以上 |
| ストレージ | 100GB | 200GB以上 |

### ソフトウェア要件

```bash
# Python 3.12.7
uv run python --version

# CUDA 12.x
nvcc --version

# 依存関係インストール
uv sync

# 追加依存関係（ラベリング用）
uv add praat-parselmouth librosa

# WandB ログイン（必須）
wandb login
```

**注意**: 学習時のログ記録には WandB が必須です。アカウントを作成し、ログインしてください。

### ベースモデルのダウンロード

```bash
mkdir -p checkpoints
huggingface-cli download SWivid/F5-TTS \
    F5TTS_v1_Base/model_1200000.safetensors \
    --local-dir checkpoints/
```

---

## 3. データセットの準備

### 3.1 論文再現用データセット（必須）

ReStyle-TTS論文（arXiv:2601.03632）を再現するには、以下の **英語データセット** を使用します。

| データセット | サイズ | 用途 | ダウンロード |
|-------------|--------|------|-------------|
| **LibriTTS** | ~585時間 | ピッチ/エネルギー | [OpenSLR](https://openslr.org/60/) |
| **VCTK** | 110話者 | ピッチ/エネルギー | [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/2950) |
| **TESS** | 2,800ファイル | 感情（7カテゴリ） | [Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) |

**重要**: 論文では **英語のみ** で学習・評価されています。

### 3.2 データセットのダウンロード

```bash
mkdir -p raw_datasets
cd raw_datasets

# ========================================
# LibriTTS (約60GB)
# ========================================
wget https://www.openslr.org/resources/60/train-clean-100.tar.gz
wget https://www.openslr.org/resources/60/train-clean-360.tar.gz
wget https://www.openslr.org/resources/60/train-other-500.tar.gz
tar -xzf train-clean-100.tar.gz
tar -xzf train-clean-360.tar.gz
tar -xzf train-other-500.tar.gz

# ========================================
# VCTK (約11GB)
# ========================================
wget https://datashare.ed.ac.uk/download/DS_10283_2950.zip
unzip DS_10283_2950.zip

# ========================================
# TESS (約200MB)
# ========================================
# Kaggle APIを使用する場合:
kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess
unzip toronto-emotional-speech-set-tess.zip -d TESS

# または手動でKaggleからダウンロード:
# https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

cd ..
```

### 3.3 データセットの前処理

```bash
# ========================================
# LibriTTS → Arrow形式
# ========================================
uv run python -m f5_tts.train.datasets.prepare_libritts

# 出力: data/LibriTTS_100_360_500_char/

# ========================================
# VCTK → Arrow形式
# ========================================
uv run python -m f5_tts.train.datasets.prepare_vctk \
    --dataset-dir raw_datasets/VCTK-Corpus-0.92

# 出力: data/VCTK_char/

# ========================================
# TESS → 感情別Arrow形式
# ========================================
uv run python -m f5_tts.train.datasets.prepare_tess \
    --dataset-dir raw_datasets/TESS

# 出力:
#   data/TESS_angry_char/
#   data/TESS_happy_char/
#   data/TESS_sad_char/
#   data/TESS_fear_char/
#   data/TESS_disgusted_char/
#   data/TESS_surprised_char/
#   data/TESS_neutral_char/
```

### 3.4 ピッチ/エネルギーラベリング

```bash
# LibriTTS + VCTKを統合してピッチ/エネルギーでラベリング
uv run python scripts/label_prosody.py \
    --input-dirs data/LibriTTS_100_360_500_char data/VCTK_char \
    --output-dir data \
    --percentile 20

# 出力:
#   data/pitch_high_char/
#   data/pitch_low_char/
#   data/energy_high_char/
#   data/energy_low_char/
```

### 3.5 データセット構造の確認

前処理後、以下のディレクトリ構造になっているか確認してください：

```
data/
├── pitch_high_char/
│   ├── raw.arrow
│   ├── duration.json
│   └── vocab.txt
├── pitch_low_char/
├── energy_high_char/
├── energy_low_char/
├── TESS_angry_char/
├── TESS_happy_char/
├── TESS_sad_char/
├── TESS_fear_char/
├── TESS_disgusted_char/
└── TESS_surprised_char/
```

---

## 4. 学習の実行

### 4.1 accelerate設定（マルチGPU用）

```bash
accelerate config
```

設定例（T4 x 3）：
- Distributed training: multi-GPU
- Number of GPUs: 3
- Mixed precision: bf16 または fp16

### 4.2 シングルGPU学習

```bash
uv run python -m f5_tts.train.train_style_lora \
    --config-name ReStyleTTS_Base \
    style_attribute=pitch_high \
    pretrained_checkpoint=checkpoints/F5TTS_v1_Base/model_1200000.safetensors \
    datasets.name=pitch_high_char
```

### 4.3 マルチGPU学習（T4 x 3）

```bash
accelerate launch --multi_gpu --num_processes 3 \
    -m f5_tts.train.train_style_lora \
    --config-name ReStyleTTS_Base \
    style_attribute=pitch_high \
    pretrained_checkpoint=checkpoints/F5TTS_v1_Base/model_1200000.safetensors \
    datasets.name=pitch_high_char
```

### 4.4 全スタイル一括学習

```bash
# 全スタイル（韻律 + 感情）を順番に学習
bash scripts/train_all_styles.sh

# 韻律属性のみ
bash scripts/train_all_styles.sh prosody

# 感情属性のみ
bash scripts/train_all_styles.sh emotion
```

### 4.5 主要パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `style_attribute` | (必須) | 学習するスタイル属性 |
| `pretrained_checkpoint` | (必須) | ベースモデルのパス |
| `datasets.name` | (必須) | データセット名 |
| `optim.epochs` | 11 | エポック数 |
| `optim.lora_learning_rate` | 1e-5 | 学習率 |
| `lora.rank` | 32 | LoRAランク |
| `lora.alpha` | 64 | LoRAスケーリング係数 |
| `datasets.batch_size_per_gpu` | 38400 | GPU当たりのバッチサイズ |

### 4.6 T4 GPU向け調整

T4 GPU（16GB VRAM）の場合、メモリ不足を防ぐため：

```bash
accelerate launch --multi_gpu --num_processes 3 \
    -m f5_tts.train.train_style_lora \
    --config-name ReStyleTTS_Base \
    style_attribute=pitch_high \
    pretrained_checkpoint=checkpoints/F5TTS_v1_Base/model_1200000.safetensors \
    datasets.name=pitch_high_char \
    datasets.batch_size_per_gpu=19200
```

### 4.7 チェックポイント

学習済みモデルは以下に保存されます：

```
ckpts/ReStyleTTS_Base_vocos_pinyin_pitch_high_char/
├── lora_pitch_high_10000.safetensors
├── lora_pitch_high_20000.safetensors
├── lora_pitch_high_last.safetensors
└── ...
```

### 4.8 論文再現設定

ReStyle-TTS論文では、各スタイル属性につき **250時間** の学習を実施しています。

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
manager.load_lora("pitch_high", "ckpts/ReStyleTTS_Base_vocos_pinyin_pitch_high_char/lora_pitch_high_last.safetensors")
manager.load_lora("angry", "ckpts/ReStyleTTS_Base_vocos_pinyin_TESS_angry_char/lora_angry_last.safetensors")

# スタイル適用して推論
with manager.apply_styles({"pitch_high": 1.0, "angry": 0.5}, use_olora=True):
    audio, sr, _ = tts.infer(
        ref_file="reference.wav",
        ref_text="参照テキスト",
        gen_text="生成したいテキスト",
        use_dcfg=True,
        lambda_t=2.0,
        lambda_a=0.5,
    )
```

### 5.2 Gradio UI

```bash
uv run python src/f5_tts/infer/infer_gradio.py
```

ブラウザで「ReStyle設定」セクションを展開して、スタイルを制御できます。

---

## 6. トラブルシューティング

### CUDA out of memory

```
解決策:
1. batch_size_per_gpu を減らす
   datasets.batch_size_per_gpu=19200
2. gradient_accumulation_steps を増やす
   optim.grad_accumulation_steps=2
3. lora.rank を減らす
   lora.rank=16
```

### データセットが見つからない

```
解決策:
1. datasets.name が正しいか確認
2. data/ディレクトリにArrow形式のデータセットがあるか確認
3. vocab.txt, duration.json, raw.arrow が存在するか確認
```

### 学習が収束しない

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
tensorboard --logdir ckpts/ReStyleTTS_Base_*/
```

---

## クイックスタート（全コマンド）

以下のコマンドを順番に実行すると、環境構築から学習開始まで完了します。

```bash
# ========================================
# Step 0: リポジトリクローン & 環境構築
# ========================================
git clone https://github.com/ayutaz/F5-TTS.git
cd F5-TTS
git checkout feature/restyle-dcfg
uv sync
uv add praat-parselmouth librosa
wandb login

# ========================================
# Step 1: ベースモデルダウンロード
# ========================================
mkdir -p checkpoints
huggingface-cli download SWivid/F5-TTS \
    F5TTS_v1_Base/model_1200000.safetensors \
    --local-dir checkpoints/

# ========================================
# Step 2: データセットダウンロード
# ========================================
mkdir -p raw_datasets && cd raw_datasets

# LibriTTS
wget https://www.openslr.org/resources/60/train-clean-100.tar.gz
wget https://www.openslr.org/resources/60/train-clean-360.tar.gz
wget https://www.openslr.org/resources/60/train-other-500.tar.gz
tar -xzf train-clean-100.tar.gz
tar -xzf train-clean-360.tar.gz
tar -xzf train-other-500.tar.gz

# VCTK
wget https://datashare.ed.ac.uk/download/DS_10283_2950.zip
unzip DS_10283_2950.zip

# TESS (Kaggleから手動ダウンロード)
# https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
unzip toronto-emotional-speech-set-tess.zip -d TESS

cd ..

# ========================================
# Step 3: データセット前処理
# ========================================
uv run python -m f5_tts.train.datasets.prepare_libritts
uv run python -m f5_tts.train.datasets.prepare_vctk --dataset-dir raw_datasets/VCTK-Corpus-0.92
uv run python -m f5_tts.train.datasets.prepare_tess --dataset-dir raw_datasets/TESS

# ========================================
# Step 4: ピッチ/エネルギーラベリング
# ========================================
uv run python scripts/label_prosody.py \
    --input-dirs data/LibriTTS_100_360_500_char data/VCTK_char \
    --output-dir data \
    --percentile 20

# ========================================
# Step 5: accelerate設定
# ========================================
accelerate config
# → multi-GPU, 3 GPUs, bf16/fp16

# ========================================
# Step 6: 学習開始
# ========================================
bash scripts/train_all_styles.sh
```

---

## 参考リンク

- [F5-TTS 公式リポジトリ](https://github.com/SWivid/F5-TTS)
- [ReStyle-TTS 論文](https://arxiv.org/abs/2601.03632)
- [LoRA 論文](https://arxiv.org/abs/2106.09685)
- [PEFT ライブラリ](https://github.com/huggingface/peft)
- [LibriTTS データセット](https://openslr.org/60/)
- [VCTK データセット](https://datashare.ed.ac.uk/handle/10283/2950)
- [TESS データセット](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
