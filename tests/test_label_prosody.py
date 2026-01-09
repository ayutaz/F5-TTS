"""
ピッチ/エネルギーラベリングスクリプトのテスト
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# scripts/label_prosody.pyをインポートするためにパスを追加
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from label_prosody import (
    compute_pitch_mean,
    compute_energy_rms,
    process_sample,
)


class TestComputePitchMean:
    """compute_pitch_mean関数のテスト"""

    @patch("label_prosody.parselmouth.Sound")
    def test_returns_float_for_valid_audio(self, mock_sound):
        """有効な音声で浮動小数点数を返す"""
        # モック設定
        mock_pitch = MagicMock()
        mock_pitch.selected_array = {"frequency": np.array([150.0, 160.0, 170.0, 0.0, 180.0])}
        mock_snd = MagicMock()
        mock_snd.to_pitch.return_value = mock_pitch
        mock_sound.return_value = mock_snd

        result = compute_pitch_mean("dummy.wav")

        assert isinstance(result, float)
        # 無声区間(0.0)を除いた平均: (150+160+170+180)/4 = 165
        assert result == pytest.approx(165.0)

    @patch("label_prosody.parselmouth.Sound")
    def test_excludes_unvoiced_frames(self, mock_sound):
        """無声区間（0Hz）を除外"""
        mock_pitch = MagicMock()
        mock_pitch.selected_array = {"frequency": np.array([0.0, 100.0, 0.0, 200.0, 0.0])}
        mock_snd = MagicMock()
        mock_snd.to_pitch.return_value = mock_pitch
        mock_sound.return_value = mock_snd

        result = compute_pitch_mean("dummy.wav")

        # 有声フレームのみ: (100+200)/2 = 150
        assert result == pytest.approx(150.0)

    @patch("label_prosody.parselmouth.Sound")
    def test_returns_none_for_all_unvoiced(self, mock_sound):
        """全て無声の場合はNone"""
        mock_pitch = MagicMock()
        mock_pitch.selected_array = {"frequency": np.array([0.0, 0.0, 0.0])}
        mock_snd = MagicMock()
        mock_snd.to_pitch.return_value = mock_pitch
        mock_sound.return_value = mock_snd

        result = compute_pitch_mean("dummy.wav")

        assert result is None

    @patch("label_prosody.parselmouth.Sound")
    def test_returns_none_on_exception(self, mock_sound):
        """例外発生時はNone"""
        mock_sound.side_effect = Exception("File not found")

        result = compute_pitch_mean("nonexistent.wav")

        assert result is None


class TestComputeEnergyRms:
    """compute_energy_rms関数のテスト"""

    @patch("label_prosody.librosa.load")
    def test_returns_float_for_valid_audio(self, mock_load):
        """有効な音声で浮動小数点数を返す"""
        # モック設定
        mock_load.return_value = (np.array([0.1, 0.2, 0.3, 0.4]), 24000)

        result = compute_energy_rms("dummy.wav")

        assert isinstance(result, float)
        # RMS = sqrt(mean([0.01, 0.04, 0.09, 0.16])) = sqrt(0.075)
        expected = np.sqrt(np.mean(np.array([0.1, 0.2, 0.3, 0.4])**2))
        assert result == pytest.approx(expected)

    @patch("label_prosody.librosa.load")
    def test_uses_target_sample_rate(self, mock_load):
        """ターゲットサンプルレートを使用"""
        mock_load.return_value = (np.array([0.1]), 24000)

        compute_energy_rms("dummy.wav", target_sr=16000)

        mock_load.assert_called_once_with("dummy.wav", sr=16000)

    @patch("label_prosody.librosa.load")
    def test_returns_none_on_exception(self, mock_load):
        """例外発生時はNone"""
        mock_load.side_effect = Exception("File not found")

        result = compute_energy_rms("nonexistent.wav")

        assert result is None


class TestProcessSample:
    """process_sample関数のテスト"""

    @patch("label_prosody.compute_energy_rms")
    @patch("label_prosody.compute_pitch_mean")
    def test_returns_dict_for_valid_sample(self, mock_pitch, mock_energy):
        """有効なサンプルで辞書を返す"""
        mock_pitch.return_value = 150.0
        mock_energy.return_value = 0.1

        args = (0, "path/to/audio.wav", "hello", 2.5)
        result = process_sample(args)

        assert result is not None
        assert result["idx"] == 0
        assert result["audio_path"] == "path/to/audio.wav"
        assert result["text"] == "hello"
        assert result["duration"] == 2.5
        assert result["pitch"] == 150.0
        assert result["energy"] == 0.1

    @patch("label_prosody.compute_energy_rms")
    @patch("label_prosody.compute_pitch_mean")
    def test_returns_none_when_pitch_fails(self, mock_pitch, mock_energy):
        """ピッチ計算失敗時はNone"""
        mock_pitch.return_value = None
        mock_energy.return_value = 0.1

        args = (0, "path/to/audio.wav", "hello", 2.5)
        result = process_sample(args)

        assert result is None

    @patch("label_prosody.compute_energy_rms")
    @patch("label_prosody.compute_pitch_mean")
    def test_returns_none_when_energy_fails(self, mock_pitch, mock_energy):
        """エネルギー計算失敗時はNone"""
        mock_pitch.return_value = 150.0
        mock_energy.return_value = None

        args = (0, "path/to/audio.wav", "hello", 2.5)
        result = process_sample(args)

        assert result is None


class TestPercentileClassification:
    """パーセンタイル分類ロジックのテスト"""

    def test_top_20_percent(self):
        """上位20%の閾値計算"""
        values = list(range(1, 101))  # 1-100
        threshold = np.percentile(values, 80)  # 上位20% = 80パーセンタイル以上

        # 80パーセンタイル = 80.2（補間により）
        high_values = [v for v in values if v >= threshold]

        # 上位20%は約20個
        assert len(high_values) == pytest.approx(20, abs=1)

    def test_bottom_20_percent(self):
        """下位20%の閾値計算"""
        values = list(range(1, 101))  # 1-100
        threshold = np.percentile(values, 20)  # 下位20% = 20パーセンタイル以下

        low_values = [v for v in values if v <= threshold]

        # 下位20%は約20個
        assert len(low_values) == pytest.approx(20, abs=1)

    def test_percentile_with_float_values(self):
        """浮動小数点値でのパーセンタイル計算"""
        pitches = [100.5, 150.2, 200.1, 250.8, 300.3]

        high_threshold = np.percentile(pitches, 80)
        low_threshold = np.percentile(pitches, 20)

        high = [p for p in pitches if p >= high_threshold]
        low = [p for p in pitches if p <= low_threshold]

        # 各カテゴリに少なくとも1つはある
        assert len(high) >= 1
        assert len(low) >= 1

    def test_classification_is_exclusive(self):
        """high/lowの分類が重複しない"""
        values = list(range(1, 101))

        high_threshold = np.percentile(values, 80)
        low_threshold = np.percentile(values, 20)

        high = [v for v in values if v >= high_threshold]
        low = [v for v in values if v <= low_threshold]

        # highとlowに重複がない
        overlap = set(high) & set(low)
        assert len(overlap) == 0


class TestSaveLabeledDataset:
    """save_labeled_dataset関数のテスト"""

    def test_empty_samples_warning(self, tmp_path, capsys):
        """空のサンプルリストで警告"""
        from label_prosody import save_labeled_dataset

        save_labeled_dataset([], tmp_path / "test", "test_label")

        captured = capsys.readouterr()
        assert "Warning" in captured.out or not (tmp_path / "test" / "raw.arrow").exists()

    @patch("label_prosody.ArrowWriter")
    def test_creates_output_files(self, mock_arrow_writer, tmp_path):
        """出力ファイルを作成"""
        from label_prosody import save_labeled_dataset

        samples = [
            {"audio_path": "/path/to/audio.wav", "text": "hello", "duration": 2.0},
        ]

        save_labeled_dataset(samples, tmp_path / "output", "test_label")

        # duration.jsonとvocab.txtが作成される
        assert (tmp_path / "output" / "duration.json").exists()
        assert (tmp_path / "output" / "vocab.txt").exists()

    @patch("label_prosody.ArrowWriter")
    def test_vocab_has_space_at_front(self, mock_arrow_writer, tmp_path):
        """vocab.txtの先頭にスペースがある"""
        from label_prosody import save_labeled_dataset

        samples = [
            {"audio_path": "/path/to/audio.wav", "text": "ab", "duration": 2.0},
        ]

        save_labeled_dataset(samples, tmp_path / "output", "test_label")

        vocab_content = (tmp_path / "output" / "vocab.txt").read_text(encoding="utf-8")
        # 各行を分割（strip()は使わない）
        lines = vocab_content.split("\n")

        # 先頭行がスペース（改行のみの行ではなく、スペース1文字）
        assert lines[0] == " "
