"""
VCTK準備スクリプトのテスト
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from f5_tts.train.datasets.prepare_vctk import (
    get_text_for_audio,
    process_speaker_dir,
)


class TestGetTextForAudio:
    """get_text_for_audio関数のテスト"""

    def test_extracts_text_file_path_mic1(self, tmp_path):
        """mic1ファイルから正しいテキストファイルパスを特定"""
        # テスト用ディレクトリ構造を作成
        txt_base = tmp_path / "txt"
        speaker_dir = txt_base / "p225"
        speaker_dir.mkdir(parents=True)

        # テキストファイル作成
        txt_file = speaker_dir / "p225_001.txt"
        txt_file.write_text("Hello world", encoding="utf-8")

        # 音声ファイルパス（実際のファイルは不要）
        audio_path = tmp_path / "wav" / "p225" / "p225_001_mic1.flac"
        audio_path.parent.mkdir(parents=True)

        result = get_text_for_audio(audio_path, txt_base)
        assert result == "Hello world"

    def test_extracts_text_file_path_mic2(self, tmp_path):
        """mic2ファイルからも正しいテキストファイルパスを特定"""
        txt_base = tmp_path / "txt"
        speaker_dir = txt_base / "p226"
        speaker_dir.mkdir(parents=True)

        txt_file = speaker_dir / "p226_002.txt"
        txt_file.write_text("Test sentence", encoding="utf-8")

        audio_path = tmp_path / "wav" / "p226" / "p226_002_mic2.flac"
        audio_path.parent.mkdir(parents=True)

        result = get_text_for_audio(audio_path, txt_base)
        assert result == "Test sentence"

    def test_returns_none_when_text_missing(self, tmp_path):
        """テキストファイルがない場合はNoneを返す"""
        txt_base = tmp_path / "txt"
        txt_base.mkdir(parents=True)

        audio_path = tmp_path / "wav" / "p999" / "p999_001_mic1.flac"

        result = get_text_for_audio(audio_path, txt_base)
        assert result is None

    def test_strips_whitespace(self, tmp_path):
        """テキストの前後の空白を除去"""
        txt_base = tmp_path / "txt"
        speaker_dir = txt_base / "p225"
        speaker_dir.mkdir(parents=True)

        txt_file = speaker_dir / "p225_003.txt"
        txt_file.write_text("  Sentence with spaces  \n", encoding="utf-8")

        audio_path = tmp_path / "wav" / "p225" / "p225_003_mic1.flac"
        audio_path.parent.mkdir(parents=True)

        result = get_text_for_audio(audio_path, txt_base)
        assert result == "Sentence with spaces"


class TestProcessSpeakerDir:
    """process_speaker_dir関数のテスト"""

    @patch("f5_tts.train.datasets.prepare_vctk.sf.info")
    def test_filters_mic1_only(self, mock_sf_info, tmp_path):
        """mic1のファイルのみを処理し、mic2は除外"""
        # モック設定
        mock_info = MagicMock()
        mock_info.duration = 5.0
        mock_sf_info.return_value = mock_info

        # ディレクトリ構造作成
        speaker_dir = tmp_path / "wav" / "p225"
        speaker_dir.mkdir(parents=True)
        txt_base = tmp_path / "txt"
        txt_speaker = txt_base / "p225"
        txt_speaker.mkdir(parents=True)

        # mic1とmic2のファイルを作成
        (speaker_dir / "p225_001_mic1.flac").touch()
        (speaker_dir / "p225_001_mic2.flac").touch()
        (speaker_dir / "p225_002_mic1.flac").touch()
        (speaker_dir / "p225_002_mic2.flac").touch()

        # テキストファイル作成
        (txt_speaker / "p225_001.txt").write_text("Text one", encoding="utf-8")
        (txt_speaker / "p225_002.txt").write_text("Text two", encoding="utf-8")

        results, durations, vocab = process_speaker_dir(speaker_dir, txt_base)

        # mic1のみ処理されている（2件）
        assert len(results) == 2
        assert len(durations) == 2

    @patch("f5_tts.train.datasets.prepare_vctk.sf.info")
    def test_filters_by_duration_too_short(self, mock_sf_info, tmp_path):
        """0.4秒未満の音声を除外"""
        mock_info = MagicMock()
        mock_info.duration = 0.3  # 0.4秒未満
        mock_sf_info.return_value = mock_info

        speaker_dir = tmp_path / "wav" / "p225"
        speaker_dir.mkdir(parents=True)
        txt_base = tmp_path / "txt"
        txt_speaker = txt_base / "p225"
        txt_speaker.mkdir(parents=True)

        (speaker_dir / "p225_001_mic1.flac").touch()
        (txt_speaker / "p225_001.txt").write_text("Short", encoding="utf-8")

        results, durations, vocab = process_speaker_dir(speaker_dir, txt_base)

        assert len(results) == 0

    @patch("f5_tts.train.datasets.prepare_vctk.sf.info")
    def test_filters_by_duration_too_long(self, mock_sf_info, tmp_path):
        """30秒超の音声を除外"""
        mock_info = MagicMock()
        mock_info.duration = 35.0  # 30秒超
        mock_sf_info.return_value = mock_info

        speaker_dir = tmp_path / "wav" / "p225"
        speaker_dir.mkdir(parents=True)
        txt_base = tmp_path / "txt"
        txt_speaker = txt_base / "p225"
        txt_speaker.mkdir(parents=True)

        (speaker_dir / "p225_001_mic1.flac").touch()
        (txt_speaker / "p225_001.txt").write_text("Long audio", encoding="utf-8")

        results, durations, vocab = process_speaker_dir(speaker_dir, txt_base)

        assert len(results) == 0

    @patch("f5_tts.train.datasets.prepare_vctk.sf.info")
    def test_valid_duration_range(self, mock_sf_info, tmp_path):
        """0.4-30秒の範囲内の音声を処理"""
        mock_info = MagicMock()
        mock_info.duration = 5.0  # 有効範囲
        mock_sf_info.return_value = mock_info

        speaker_dir = tmp_path / "wav" / "p225"
        speaker_dir.mkdir(parents=True)
        txt_base = tmp_path / "txt"
        txt_speaker = txt_base / "p225"
        txt_speaker.mkdir(parents=True)

        (speaker_dir / "p225_001_mic1.flac").touch()
        (txt_speaker / "p225_001.txt").write_text("Valid audio", encoding="utf-8")

        results, durations, vocab = process_speaker_dir(speaker_dir, txt_base)

        assert len(results) == 1
        assert results[0]["text"] == "Valid audio"
        assert results[0]["duration"] == 5.0

    @patch("f5_tts.train.datasets.prepare_vctk.sf.info")
    def test_collects_vocabulary(self, mock_sf_info, tmp_path):
        """語彙セットを正しく収集"""
        mock_info = MagicMock()
        mock_info.duration = 5.0
        mock_sf_info.return_value = mock_info

        speaker_dir = tmp_path / "wav" / "p225"
        speaker_dir.mkdir(parents=True)
        txt_base = tmp_path / "txt"
        txt_speaker = txt_base / "p225"
        txt_speaker.mkdir(parents=True)

        (speaker_dir / "p225_001_mic1.flac").touch()
        (txt_speaker / "p225_001.txt").write_text("abc", encoding="utf-8")

        results, durations, vocab = process_speaker_dir(speaker_dir, txt_base)

        assert "a" in vocab
        assert "b" in vocab
        assert "c" in vocab

    @patch("f5_tts.train.datasets.prepare_vctk.sf.info")
    def test_skips_missing_text(self, mock_sf_info, tmp_path):
        """テキストがない音声ファイルをスキップ"""
        mock_info = MagicMock()
        mock_info.duration = 5.0
        mock_sf_info.return_value = mock_info

        speaker_dir = tmp_path / "wav" / "p225"
        speaker_dir.mkdir(parents=True)
        txt_base = tmp_path / "txt"
        txt_speaker = txt_base / "p225"
        txt_speaker.mkdir(parents=True)

        # 音声ファイルはあるがテキストがない
        (speaker_dir / "p225_001_mic1.flac").touch()
        # テキストファイルは作成しない

        results, durations, vocab = process_speaker_dir(speaker_dir, txt_base)

        assert len(results) == 0
