#!/bin/bash
#
# ReStyle-TTS 全スタイル一括学習スクリプト
#
# このスクリプトは全てのStyle LoRA（韻律4種 + 感情6種）を順番に学習します。
# T4 GPU x 3 向けに設定されています。
#
# 使用法:
#   # 全スタイルを学習
#   bash scripts/train_all_styles.sh
#
#   # 韻律属性のみ学習
#   bash scripts/train_all_styles.sh prosody
#
#   # 感情属性のみ学習
#   bash scripts/train_all_styles.sh emotion
#
# 前提条件:
#   - データセットが準備済み（data/pitch_high_char, data/TESS_angry_char 等）
#   - ベースモデルがダウンロード済み
#   - accelerate が設定済み（accelerate config）
#   - wandb にログイン済み
#

set -e  # エラー時に停止

# ========================================
# 設定
# ========================================
CHECKPOINT="checkpoints/F5TTS_v1_Base/model_1200000.safetensors"
NUM_GPUS=3
CONFIG_NAME="ReStyleTTS_Base"

# 韻律属性（LibriTTS + VCTK由来）
PROSODY_STYLES=("pitch_high" "pitch_low" "energy_high" "energy_low")

# 感情属性（TESS由来）
EMOTION_STYLES=("angry" "happy" "sad" "fear" "disgusted" "surprised")

# ========================================
# チェック
# ========================================
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: ベースモデルが見つかりません: $CHECKPOINT"
    echo "以下のコマンドでダウンロードしてください:"
    echo "  huggingface-cli download SWivid/F5-TTS F5TTS_v1_Base/model_1200000.safetensors --local-dir checkpoints/"
    exit 1
fi

# ========================================
# 学習関数
# ========================================
train_style() {
    local style=$1
    local dataset=$2

    echo ""
    echo "========================================"
    echo "学習開始: $style"
    echo "データセット: $dataset"
    echo "========================================"

    # データセット存在チェック
    if [ ! -d "data/$dataset" ]; then
        echo "Warning: データセットが見つかりません: data/$dataset"
        echo "スキップします..."
        return
    fi

    accelerate launch --multi_gpu --num_processes $NUM_GPUS \
        -m f5_tts.train.train_style_lora \
        --config-name $CONFIG_NAME \
        style_attribute=$style \
        pretrained_checkpoint=$CHECKPOINT \
        datasets.name=$dataset

    echo "$style の学習が完了しました"
}

# ========================================
# メイン処理
# ========================================
MODE=${1:-all}

echo "ReStyle-TTS 全スタイル学習"
echo "モード: $MODE"
echo "GPU数: $NUM_GPUS"
echo ""

# 韻律属性の学習
if [ "$MODE" = "all" ] || [ "$MODE" = "prosody" ]; then
    echo "=== 韻律属性の学習 ==="
    for style in "${PROSODY_STYLES[@]}"; do
        train_style "$style" "${style}_char"
    done
fi

# 感情属性の学習
if [ "$MODE" = "all" ] || [ "$MODE" = "emotion" ]; then
    echo "=== 感情属性の学習 ==="
    for style in "${EMOTION_STYLES[@]}"; do
        train_style "$style" "TESS_${style}_char"
    done
fi

echo ""
echo "========================================"
echo "全ての学習が完了しました"
echo "========================================"
echo ""
echo "学習済みLoRAは以下に保存されています:"
echo "  ckpts/ReStyleTTS_Base_*/lora_*_last.safetensors"
