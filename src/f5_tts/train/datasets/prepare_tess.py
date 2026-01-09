"""
TESS (Toronto Emotional Speech Set) 準備スクリプト

TESSは感情音声データセットで、ReStyle-TTS論文では感情Style LoRAの学習に使用。
7つの感情カテゴリ（angry, disgust, fear, happy, neutral, sad, surprised）を含む。

ダウンロード:
    https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
    または
    kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess

解凍後の構造:
    TESS/
    ├── OAF_angry/
    │   ├── OAF_back_angry.wav
    │   ├── OAF_bar_angry.wav
    │   └── ...
    ├── OAF_disgust/
    ├── OAF_fear/
    ├── OAF_happy/
    ├── OAF_neutral/
    ├── OAF_pleasant_surprised/  (→ surprised として処理)
    ├── OAF_sad/
    ├── YAF_angry/
    ├── YAF_disgust/
    └── ...

使用法:
    # 全感情を個別データセットとして準備
    uv run python -m f5_tts.train.datasets.prepare_tess

    # 特定の感情のみ準備
    uv run python -m f5_tts.train.datasets.prepare_tess --emotions angry happy

    # カスタムパスを指定
    uv run python -m f5_tts.train.datasets.prepare_tess \
        --dataset-dir /path/to/TESS \
        --save-dir /path/to/output

出力:
    data/TESS_angry_char/
    data/TESS_happy_char/
    data/TESS_sad_char/
    data/TESS_fear_char/
    data/TESS_disgusted_char/
    data/TESS_surprised_char/
    data/TESS_neutral_char/
"""

import argparse
import json
import os
import sys
from importlib.resources import files
from pathlib import Path

import soundfile as sf
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm


# 感情マッピング（ディレクトリ名 → スタイル属性名）
EMOTION_MAPPING = {
    "angry": "angry",
    "disgust": "disgusted",  # 論文のスタイル属性名に合わせる
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "pleasant_surprised": "surprised",  # 論文のスタイル属性名に合わせる
    "sad": "sad",
}


def extract_word_from_filename(filename: str) -> str:
    """ファイル名から単語を抽出

    例:
        OAF_back_angry.wav -> back
        YAF_bar_happy.wav -> bar
        OAF_bite_fear.wav -> bite

    Args:
        filename: 音声ファイル名

    Returns:
        抽出された単語
    """
    # 拡張子を除去
    stem = Path(filename).stem  # OAF_back_angry

    # アンダースコアで分割
    parts = stem.split("_")

    # 最初（話者ID）と最後（感情）を除いた部分が単語
    # 通常は parts[1] だが、単語に_が含まれる場合を考慮
    if len(parts) >= 3:
        word_parts = parts[1:-1]
        return "_".join(word_parts)
    elif len(parts) == 2:
        return parts[1]
    else:
        return stem


def get_emotion_from_dirname(dirname: str) -> str | None:
    """ディレクトリ名から感情を取得

    例:
        OAF_angry -> angry
        YAF_happy -> happy
        OAF_pleasant_surprised -> pleasant_surprised

    Args:
        dirname: ディレクトリ名

    Returns:
        感情名、マッチしない場合はNone
    """
    # プレフィックス（OAF_, YAF_）を除去
    if dirname.startswith(("OAF_", "YAF_")):
        emotion_part = dirname[4:]  # OAF_ or YAF_ の後ろ
        if emotion_part in EMOTION_MAPPING:
            return emotion_part
    return None


def process_emotion_dirs(
    dataset_dir: Path,
    emotion_key: str,
) -> tuple[list, list, set]:
    """特定の感情のディレクトリを処理

    Args:
        dataset_dir: TESSデータセットのベースディレクトリ
        emotion_key: 感情キー（EMOTION_MAPPINGのキー）

    Returns:
        (結果リスト, 長さリスト, 語彙セット)
    """
    results = []
    durations = []
    vocab_set = set()

    # OAF_* と YAF_* の両方を処理
    for prefix in ["OAF", "YAF"]:
        emotion_dir = dataset_dir / f"{prefix}_{emotion_key}"

        if not emotion_dir.exists():
            continue

        for audio_path in emotion_dir.glob("*.wav"):
            try:
                # 単語を抽出
                word = extract_word_from_filename(audio_path.name)

                # 長さチェック
                info = sf.info(audio_path)
                duration = info.duration

                # TESSは短い発話なので0.3秒〜5秒で制限
                if duration < 0.3 or duration > 5:
                    continue

                results.append({
                    "audio_path": str(audio_path.absolute()),
                    "text": word,
                    "duration": duration,
                })
                durations.append(duration)
                vocab_set.update(list(word))

            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue

    return results, durations, vocab_set


def save_dataset(
    results: list,
    durations: list,
    vocab_set: set,
    save_dir: Path,
    emotion_name: str,
):
    """データセットを保存

    Args:
        results: 結果リスト
        durations: 長さリスト
        vocab_set: 語彙セット
        save_dir: 保存先ディレクトリ
        emotion_name: 感情名
    """
    if not results:
        print(f"  Warning: {emotion_name}のサンプルが見つかりませんでした")
        return

    # 出力ディレクトリ作成
    save_dir.mkdir(parents=True, exist_ok=True)

    # Arrow形式で保存
    with ArrowWriter(path=str(save_dir / "raw.arrow")) as writer:
        for item in results:
            writer.write(item)
        writer.finalize()

    # duration.json保存
    with open(save_dir / "duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": durations}, f, ensure_ascii=False)

    # vocab.txt保存（スペースを先頭に）
    vocab_list = sorted(vocab_set)
    if " " in vocab_list:
        vocab_list.remove(" ")
    vocab_list = [" "] + vocab_list

    with open(save_dir / "vocab.txt", "w", encoding="utf-8") as f:
        for char in vocab_list:
            f.write(char + "\n")

    total_seconds = sum(durations)
    print(f"  {emotion_name}: {len(results)}サンプル, {total_seconds:.1f}秒")


def main():
    parser = argparse.ArgumentParser(description="TESS準備スクリプト")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="raw_datasets/TESS",
        help="TESSデータセットのディレクトリ",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="出力ディレクトリのベースパス（デフォルト: data/）",
    )
    parser.add_argument(
        "--emotions",
        nargs="+",
        choices=list(EMOTION_MAPPING.values()),
        default=None,
        help="処理する感情のリスト（デフォルト: 全感情）",
    )
    args = parser.parse_args()

    # パス設定
    dataset_dir = Path(args.dataset_dir)

    if args.save_dir:
        base_save_dir = Path(args.save_dir)
    else:
        base_save_dir = Path(str(files("f5_tts").joinpath("../../data")))

    # ディレクトリチェック
    if not dataset_dir.exists():
        print(f"Error: データセットディレクトリが見つかりません: {dataset_dir}")
        print("TESSデータセットをダウンロードして解凍してください。")
        print("  https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess")
        sys.exit(1)

    # 処理する感情を決定
    if args.emotions:
        # 指定された感情のみ
        target_emotions = {
            k: v for k, v in EMOTION_MAPPING.items()
            if v in args.emotions
        }
    else:
        # 全感情
        target_emotions = EMOTION_MAPPING

    print(f"TESS準備")
    print(f"  データセット: {dataset_dir}")
    print(f"  出力先: {base_save_dir}")
    print(f"  感情: {list(target_emotions.values())}")
    print()

    # 各感情を処理
    for emotion_key, emotion_name in tqdm(target_emotions.items(), desc="Processing emotions"):
        results, durations, vocab_set = process_emotion_dirs(dataset_dir, emotion_key)

        save_dir = base_save_dir / f"TESS_{emotion_name}_char"
        save_dataset(results, durations, vocab_set, save_dir, emotion_name)

    print(f"\n=== 完了 ===")
    print(f"出力先: {base_save_dir}/TESS_*_char/")


if __name__ == "__main__":
    main()
