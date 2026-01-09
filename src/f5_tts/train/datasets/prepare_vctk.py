"""
VCTK Corpus準備スクリプト

CSTR VCTK Corpus: 110話者の英語音声データセット
ReStyle-TTS論文ではピッチ/エネルギーのラベリングに使用。

ダウンロード:
    https://datashare.ed.ac.uk/handle/10283/2950
    または
    wget https://datashare.ed.ac.uk/download/DS_10283_2950.zip

解凍後の構造:
    VCTK-Corpus-0.92/
    ├── wav48_silence_trimmed/
    │   ├── p225/
    │   │   ├── p225_001_mic1.flac
    │   │   ├── p225_001_mic2.flac
    │   │   └── ...
    │   ├── p226/
    │   └── ...
    └── txt/
        ├── p225/
        │   ├── p225_001.txt
        │   └── ...
        └── ...

使用法:
    # デフォルト設定で実行
    uv run python -m f5_tts.train.datasets.prepare_vctk

    # カスタムパスを指定
    uv run python -m f5_tts.train.datasets.prepare_vctk \
        --dataset-dir /path/to/VCTK-Corpus-0.92 \
        --save-dir /path/to/output

出力:
    data/VCTK_char/
    ├── raw.arrow
    ├── duration.json
    └── vocab.txt
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.resources import files
from pathlib import Path

import soundfile as sf
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm


def get_text_for_audio(audio_path: Path, txt_base_dir: Path) -> str | None:
    """音声ファイルに対応するテキストを取得

    Args:
        audio_path: 音声ファイルパス (例: p225/p225_001_mic1.flac)
        txt_base_dir: テキストディレクトリのベースパス

    Returns:
        テキスト文字列、見つからない場合はNone
    """
    # 音声ファイル名からテキストファイル名を推測
    # p225_001_mic1.flac -> p225_001.txt
    speaker_id = audio_path.parent.name  # p225
    audio_stem = audio_path.stem  # p225_001_mic1
    # _mic1 or _mic2 を除去
    base_name = audio_stem.rsplit("_mic", 1)[0]  # p225_001

    txt_path = txt_base_dir / speaker_id / f"{base_name}.txt"

    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8").strip()
    return None


def process_speaker_dir(speaker_dir: Path, txt_base_dir: Path) -> tuple[list, list, set]:
    """話者ディレクトリを処理

    Args:
        speaker_dir: 話者の音声ディレクトリ
        txt_base_dir: テキストディレクトリのベースパス

    Returns:
        (結果リスト, 長さリスト, 語彙セット)
    """
    results = []
    durations = []
    vocab_set = set()

    # mic1のみを使用（mic2は除外）
    audio_files = list(speaker_dir.glob("*_mic1.flac"))

    for audio_path in audio_files:
        try:
            # テキスト取得
            text = get_text_for_audio(audio_path, txt_base_dir)
            if text is None:
                continue

            # 長さチェック
            info = sf.info(audio_path)
            duration = info.duration

            # 0.4秒〜30秒のサンプルのみ
            if duration < 0.4 or duration > 30:
                continue

            results.append({
                "audio_path": str(audio_path.absolute()),
                "text": text,
                "duration": duration,
            })
            durations.append(duration)
            vocab_set.update(list(text))

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue

    return results, durations, vocab_set


def main():
    parser = argparse.ArgumentParser(description="VCTK Corpus準備スクリプト")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="raw_datasets/VCTK-Corpus-0.92",
        help="VCTKデータセットのディレクトリ",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="出力ディレクトリ（デフォルト: data/VCTK_char）",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="並列処理のワーカー数",
    )
    args = parser.parse_args()

    # パス設定
    dataset_dir = Path(args.dataset_dir)
    wav_dir = dataset_dir / "wav48_silence_trimmed"
    txt_dir = dataset_dir / "txt"

    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = Path(str(files("f5_tts").joinpath("../../data/VCTK_char")))

    # ディレクトリチェック
    if not wav_dir.exists():
        print(f"Error: 音声ディレクトリが見つかりません: {wav_dir}")
        print("VCTKデータセットをダウンロードして解凍してください。")
        sys.exit(1)

    if not txt_dir.exists():
        print(f"Error: テキストディレクトリが見つかりません: {txt_dir}")
        sys.exit(1)

    print(f"VCTK Corpus準備")
    print(f"  データセット: {dataset_dir}")
    print(f"  出力先: {save_dir}")

    # 話者ディレクトリを取得
    speaker_dirs = sorted([d for d in wav_dir.iterdir() if d.is_dir()])
    print(f"  話者数: {len(speaker_dirs)}")

    # 並列処理で各話者を処理
    all_results = []
    all_durations = []
    all_vocab = set()

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_speaker_dir, speaker_dir, txt_dir): speaker_dir
            for speaker_dir in speaker_dirs
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing speakers"):
            results, durations, vocab = future.result()
            all_results.extend(results)
            all_durations.extend(durations)
            all_vocab.update(vocab)

    if not all_results:
        print("Error: 有効なサンプルが見つかりませんでした。")
        sys.exit(1)

    # 出力ディレクトリ作成
    save_dir.mkdir(parents=True, exist_ok=True)

    # Arrow形式で保存
    print(f"\nArrowファイルを保存中...")
    with ArrowWriter(path=str(save_dir / "raw.arrow")) as writer:
        for item in tqdm(all_results, desc="Writing to raw.arrow"):
            writer.write(item)
        writer.finalize()

    # duration.json保存
    with open(save_dir / "duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": all_durations}, f, ensure_ascii=False)

    # vocab.txt保存（スペースを先頭に）
    vocab_list = sorted(all_vocab)
    if " " in vocab_list:
        vocab_list.remove(" ")
    vocab_list = [" "] + vocab_list  # スペースを先頭に

    with open(save_dir / "vocab.txt", "w", encoding="utf-8") as f:
        for char in vocab_list:
            f.write(char + "\n")

    # 統計表示
    total_hours = sum(all_durations) / 3600
    print(f"\n=== 完了 ===")
    print(f"サンプル数: {len(all_results):,}")
    print(f"総時間: {total_hours:.2f}時間")
    print(f"語彙サイズ: {len(vocab_list)}")
    print(f"出力先: {save_dir}")


if __name__ == "__main__":
    main()
