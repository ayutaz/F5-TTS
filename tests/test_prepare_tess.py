"""
TESS準備スクリプトのテスト
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from f5_tts.train.datasets.prepare_tess import (
    extract_word_from_filename,
    get_emotion_from_dirname,
    EMOTION_MAPPING,
    process_emotion_dirs,
)


class TestExtractWordFromFilename:
    """extract_word_from_filename関数のテスト"""

    def test_standard_format_oaf(self):
        """OAF_back_angry.wav -> back"""
        result = extract_word_from_filename("OAF_back_angry.wav")
        assert result == "back"

    def test_standard_format_yaf(self):
        """YAF_bar_happy.wav -> bar"""
        result = extract_word_from_filename("YAF_bar_happy.wav")
        assert result == "bar"

    def test_different_emotion(self):
        """OAF_bite_fear.wav -> bite"""
        result = extract_word_from_filename("OAF_bite_fear.wav")
        assert result == "bite"

    def test_underscore_in_word(self):
        """単語にアンダースコアが含まれる場合"""
        # 例: OAF_some_word_angry.wav -> some_word
        result = extract_word_from_filename("OAF_some_word_angry.wav")
        assert result == "some_word"

    def test_two_parts_only(self):
        """2パートのみの場合"""
        result = extract_word_from_filename("OAF_angry.wav")
        assert result == "angry"

    def test_single_part(self):
        """1パートのみの場合"""
        result = extract_word_from_filename("word.wav")
        assert result == "word"


class TestGetEmotionFromDirname:
    """get_emotion_from_dirname関数のテスト"""

    def test_oaf_angry(self):
        """OAF_angry -> angry"""
        result = get_emotion_from_dirname("OAF_angry")
        assert result == "angry"

    def test_yaf_happy(self):
        """YAF_happy -> happy"""
        result = get_emotion_from_dirname("YAF_happy")
        assert result == "happy"

    def test_oaf_sad(self):
        """OAF_sad -> sad"""
        result = get_emotion_from_dirname("OAF_sad")
        assert result == "sad"

    def test_oaf_fear(self):
        """OAF_fear -> fear"""
        result = get_emotion_from_dirname("OAF_fear")
        assert result == "fear"

    def test_oaf_disgust(self):
        """OAF_disgust -> disgust"""
        result = get_emotion_from_dirname("OAF_disgust")
        assert result == "disgust"

    def test_oaf_neutral(self):
        """OAF_neutral -> neutral"""
        result = get_emotion_from_dirname("OAF_neutral")
        assert result == "neutral"

    def test_pleasant_surprised(self):
        """OAF_pleasant_surprised -> pleasant_surprised"""
        result = get_emotion_from_dirname("OAF_pleasant_surprised")
        assert result == "pleasant_surprised"

    def test_invalid_prefix(self):
        """無効なプレフィックスの場合はNone"""
        result = get_emotion_from_dirname("XAF_angry")
        assert result is None

    def test_no_prefix(self):
        """プレフィックスがない場合はNone"""
        result = get_emotion_from_dirname("angry")
        assert result is None

    def test_unknown_emotion(self):
        """未知の感情の場合はNone"""
        result = get_emotion_from_dirname("OAF_unknown")
        assert result is None


class TestEmotionMapping:
    """EMOTION_MAPPING定数のテスト"""

    def test_all_emotions_defined(self):
        """7感情すべてが定義されている"""
        expected_keys = {
            "angry", "disgust", "fear", "happy",
            "neutral", "pleasant_surprised", "sad"
        }
        assert set(EMOTION_MAPPING.keys()) == expected_keys

    def test_mapping_count(self):
        """7つのマッピングがある"""
        assert len(EMOTION_MAPPING) == 7

    def test_disgust_to_disgusted(self):
        """disgust -> disgusted"""
        assert EMOTION_MAPPING["disgust"] == "disgusted"

    def test_pleasant_surprised_to_surprised(self):
        """pleasant_surprised -> surprised"""
        assert EMOTION_MAPPING["pleasant_surprised"] == "surprised"

    def test_angry_unchanged(self):
        """angry -> angry"""
        assert EMOTION_MAPPING["angry"] == "angry"

    def test_happy_unchanged(self):
        """happy -> happy"""
        assert EMOTION_MAPPING["happy"] == "happy"

    def test_sad_unchanged(self):
        """sad -> sad"""
        assert EMOTION_MAPPING["sad"] == "sad"

    def test_fear_unchanged(self):
        """fear -> fear"""
        assert EMOTION_MAPPING["fear"] == "fear"

    def test_neutral_unchanged(self):
        """neutral -> neutral"""
        assert EMOTION_MAPPING["neutral"] == "neutral"


class TestProcessEmotionDirs:
    """process_emotion_dirs関数のテスト"""

    @patch("f5_tts.train.datasets.prepare_tess.sf.info")
    def test_processes_oaf_and_yaf_dirs(self, mock_sf_info, tmp_path):
        """OAFとYAFの両方のディレクトリを処理"""
        mock_info = MagicMock()
        mock_info.duration = 1.0
        mock_sf_info.return_value = mock_info

        # ディレクトリ構造作成
        oaf_dir = tmp_path / "OAF_angry"
        yaf_dir = tmp_path / "YAF_angry"
        oaf_dir.mkdir()
        yaf_dir.mkdir()

        # 音声ファイル作成
        (oaf_dir / "OAF_back_angry.wav").touch()
        (yaf_dir / "YAF_bar_angry.wav").touch()

        results, durations, vocab = process_emotion_dirs(tmp_path, "angry")

        assert len(results) == 2

    @patch("f5_tts.train.datasets.prepare_tess.sf.info")
    def test_filters_by_duration_too_short(self, mock_sf_info, tmp_path):
        """0.3秒未満の音声を除外"""
        mock_info = MagicMock()
        mock_info.duration = 0.2  # 0.3秒未満
        mock_sf_info.return_value = mock_info

        oaf_dir = tmp_path / "OAF_angry"
        oaf_dir.mkdir()
        (oaf_dir / "OAF_back_angry.wav").touch()

        results, durations, vocab = process_emotion_dirs(tmp_path, "angry")

        assert len(results) == 0

    @patch("f5_tts.train.datasets.prepare_tess.sf.info")
    def test_filters_by_duration_too_long(self, mock_sf_info, tmp_path):
        """5秒超の音声を除外"""
        mock_info = MagicMock()
        mock_info.duration = 6.0  # 5秒超
        mock_sf_info.return_value = mock_info

        oaf_dir = tmp_path / "OAF_angry"
        oaf_dir.mkdir()
        (oaf_dir / "OAF_back_angry.wav").touch()

        results, durations, vocab = process_emotion_dirs(tmp_path, "angry")

        assert len(results) == 0

    @patch("f5_tts.train.datasets.prepare_tess.sf.info")
    def test_valid_duration_range(self, mock_sf_info, tmp_path):
        """0.3-5秒の範囲内の音声を処理"""
        mock_info = MagicMock()
        mock_info.duration = 1.0  # 有効範囲
        mock_sf_info.return_value = mock_info

        oaf_dir = tmp_path / "OAF_angry"
        oaf_dir.mkdir()
        (oaf_dir / "OAF_back_angry.wav").touch()

        results, durations, vocab = process_emotion_dirs(tmp_path, "angry")

        assert len(results) == 1
        assert results[0]["text"] == "back"
        assert results[0]["duration"] == 1.0

    @patch("f5_tts.train.datasets.prepare_tess.sf.info")
    def test_extracts_word_as_text(self, mock_sf_info, tmp_path):
        """ファイル名から単語を抽出してテキストとする"""
        mock_info = MagicMock()
        mock_info.duration = 1.0
        mock_sf_info.return_value = mock_info

        oaf_dir = tmp_path / "OAF_happy"
        oaf_dir.mkdir()
        (oaf_dir / "OAF_hello_happy.wav").touch()

        results, durations, vocab = process_emotion_dirs(tmp_path, "happy")

        assert results[0]["text"] == "hello"

    @patch("f5_tts.train.datasets.prepare_tess.sf.info")
    def test_collects_vocabulary(self, mock_sf_info, tmp_path):
        """語彙セットを正しく収集"""
        mock_info = MagicMock()
        mock_info.duration = 1.0
        mock_sf_info.return_value = mock_info

        oaf_dir = tmp_path / "OAF_angry"
        oaf_dir.mkdir()
        (oaf_dir / "OAF_abc_angry.wav").touch()

        results, durations, vocab = process_emotion_dirs(tmp_path, "angry")

        assert "a" in vocab
        assert "b" in vocab
        assert "c" in vocab

    def test_nonexistent_emotion_dir(self, tmp_path):
        """感情ディレクトリが存在しない場合は空リスト"""
        results, durations, vocab = process_emotion_dirs(tmp_path, "angry")

        assert len(results) == 0
        assert len(durations) == 0
        assert len(vocab) == 0
