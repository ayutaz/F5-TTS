#!/usr/bin/env python
"""
ピッチ/エネルギー自動ラベリングスクリプト

準備済みのArrowデータセット（LibriTTS, VCTK等）からピッチとエネルギーを計算し、
上位/下位のパーセンタイルでデータを分類して新しいデータセットを作成する。

ReStyle-TTS論文では、上位/下位20%で high/low を分類。

依存関係:
    pip install parselmouth librosa

使用法:
    # LibriTTS + VCTKを統合してラベリング
    uv run python scripts/label_prosody.py \
        --input-dirs data/LibriTTS_100_360_500_char data/VCTK_char \
        --output-dir data \
        --percentile 20

    # LibriTTSのみ
    uv run python scripts/label_prosody.py \
        --input-dirs data/LibriTTS_100_360_500_char \
        --output-dir data \
        --percentile 20

出力:
    data/pitch_high_char/
    data/pitch_low_char/
    data/energy_high_char/
    data/energy_low_char/
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

# Parselmouth と librosa は遅延インポート（インストールチェック用）
try:
    import parselmouth
except ImportError:
    print("Error: parselmouth がインストールされていません")
    print("  pip install praat-parselmouth")
    sys.exit(1)

try:
    import librosa
except ImportError:
    print("Error: librosa がインストールされていません")
    print("  pip install librosa")
    sys.exit(1)


def compute_pitch_mean(audio_path: str) -> float | None:
    """音声ファイルの平均ピッチを計算

    Args:
        audio_path: 音声ファイルパス

    Returns:
        平均ピッチ（Hz）、計算できない場合はNone
    """
    try:
        snd = parselmouth.Sound(audio_path)
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array["frequency"]
        pitch_values = pitch_values[pitch_values > 0]  # 無声区間を除外

        if len(pitch_values) == 0:
            return None

        return float(np.mean(pitch_values))
    except Exception:
        return None


def compute_energy_rms(audio_path: str, target_sr: int = 24000) -> float | None:
    """音声ファイルのRMSエネルギーを計算

    Args:
        audio_path: 音声ファイルパス
        target_sr: ターゲットサンプリングレート

    Returns:
        RMSエネルギー、計算できない場合はNone
    """
    try:
        y, sr = librosa.load(audio_path, sr=target_sr)
        rms = float(np.sqrt(np.mean(y**2)))
        return rms
    except Exception:
        return None


def process_sample(args: tuple) -> dict | None:
    """サンプルを処理してピッチ/エネルギーを計算

    Args:
        args: (インデックス, audio_path, text, duration)

    Returns:
        処理結果の辞書、失敗した場合はNone
    """
    idx, audio_path, text, duration = args

    pitch = compute_pitch_mean(audio_path)
    energy = compute_energy_rms(audio_path)

    if pitch is None or energy is None:
        return None

    return {
        "idx": idx,
        "audio_path": audio_path,
        "text": text,
        "duration": duration,
        "pitch": pitch,
        "energy": energy,
    }


def load_arrow_dataset(dataset_dir: Path) -> list[dict]:
    """Arrowデータセットを読み込み

    Args:
        dataset_dir: データセットディレクトリ

    Returns:
        サンプルのリスト
    """
    # raw.arrow または raw/ ディレクトリを試行
    arrow_path = dataset_dir / "raw.arrow"
    raw_dir = dataset_dir / "raw"

    if raw_dir.exists():
        dataset = load_from_disk(str(raw_dir))
    elif arrow_path.exists():
        dataset = Dataset.from_file(str(arrow_path))
    else:
        raise FileNotFoundError(f"データセットが見つかりません: {dataset_dir}")

    return list(dataset)


def save_labeled_dataset(
    samples: list[dict],
    save_dir: Path,
    label_name: str,
):
    """ラベリング済みデータセットを保存

    Args:
        samples: サンプルリスト
        save_dir: 保存先ディレクトリ
        label_name: ラベル名（pitch_high等）
    """
    if not samples:
        print(f"  Warning: {label_name}のサンプルがありません")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    # 結果リストと語彙セット
    results = []
    durations = []
    vocab_set = set()

    for sample in samples:
        results.append({
            "audio_path": sample["audio_path"],
            "text": sample["text"],
            "duration": sample["duration"],
        })
        durations.append(sample["duration"])
        vocab_set.update(list(sample["text"]))

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

    total_hours = sum(durations) / 3600
    print(f"  {label_name}: {len(samples):,}サンプル, {total_hours:.2f}時間")


def main():
    parser = argparse.ArgumentParser(
        description="ピッチ/エネルギー自動ラベリングスクリプト"
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="入力データセットディレクトリ（複数指定可）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="出力ディレクトリのベースパス",
    )
    parser.add_argument(
        "--percentile",
        type=int,
        default=20,
        help="high/lowの閾値パーセンタイル（デフォルト: 20）",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="並列処理のワーカー数",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    percentile = args.percentile

    print("ピッチ/エネルギーラベリング")
    print(f"  入力: {args.input_dirs}")
    print(f"  出力: {output_dir}")
    print(f"  パーセンタイル: {percentile}%")
    print()

    # 全データセットを読み込み
    print("データセットを読み込み中...")
    all_samples = []
    for input_dir in args.input_dirs:
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"  Warning: {input_path}が見つかりません、スキップします")
            continue

        samples = load_arrow_dataset(input_path)
        print(f"  {input_path.name}: {len(samples):,}サンプル")
        all_samples.extend(samples)

    if not all_samples:
        print("Error: 有効なサンプルが見つかりませんでした")
        sys.exit(1)

    print(f"  合計: {len(all_samples):,}サンプル")
    print()

    # ピッチ/エネルギーを計算
    print("ピッチ/エネルギーを計算中...")
    process_args = [
        (i, s["audio_path"], s["text"], s["duration"])
        for i, s in enumerate(all_samples)
    ]

    processed_samples = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_sample, arg) for arg in process_args]

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                processed_samples.append(result)

    print(f"  有効サンプル: {len(processed_samples):,}")
    print()

    # パーセンタイルを計算
    pitches = [s["pitch"] for s in processed_samples]
    energies = [s["energy"] for s in processed_samples]

    pitch_high_threshold = np.percentile(pitches, 100 - percentile)
    pitch_low_threshold = np.percentile(pitches, percentile)
    energy_high_threshold = np.percentile(energies, 100 - percentile)
    energy_low_threshold = np.percentile(energies, percentile)

    print("閾値:")
    print(f"  ピッチ高: >= {pitch_high_threshold:.1f} Hz")
    print(f"  ピッチ低: <= {pitch_low_threshold:.1f} Hz")
    print(f"  エネルギー高: >= {energy_high_threshold:.6f}")
    print(f"  エネルギー低: <= {energy_low_threshold:.6f}")
    print()

    # 分類
    pitch_high = [s for s in processed_samples if s["pitch"] >= pitch_high_threshold]
    pitch_low = [s for s in processed_samples if s["pitch"] <= pitch_low_threshold]
    energy_high = [s for s in processed_samples if s["energy"] >= energy_high_threshold]
    energy_low = [s for s in processed_samples if s["energy"] <= energy_low_threshold]

    # 保存
    print("データセットを保存中...")
    save_labeled_dataset(pitch_high, output_dir / "pitch_high_char", "pitch_high")
    save_labeled_dataset(pitch_low, output_dir / "pitch_low_char", "pitch_low")
    save_labeled_dataset(energy_high, output_dir / "energy_high_char", "energy_high")
    save_labeled_dataset(energy_low, output_dir / "energy_low_char", "energy_low")

    print()
    print("=== 完了 ===")
    print(f"出力先: {output_dir}/{{pitch_high,pitch_low,energy_high,energy_low}}_char/")


if __name__ == "__main__":
    main()
