"""
DCFG (Decoupled Classifier-Free Guidance) のテスト

ReStyle-TTS Phase 1 の実装テスト
"""

import pytest
import torch

from f5_tts.restyle.dcfg import DCFGConfig, dcfg_combine


class TestDCFGConfig:
    """DCFGConfig のテスト"""

    def test_default_values(self):
        """デフォルト値が正しいか"""
        config = DCFGConfig()
        assert config.lambda_t == 2.0
        assert config.lambda_a == 0.5
        assert config.enabled is True

    def test_custom_values(self):
        """カスタム値が設定できるか"""
        config = DCFGConfig(lambda_t=3.0, lambda_a=0.8, enabled=False)
        assert config.lambda_t == 3.0
        assert config.lambda_a == 0.8
        assert config.enabled is False


class TestDCFGCombine:
    """dcfg_combine関数のテスト"""

    def test_basic_combination(self):
        """基本的な合成が正しく動作するか"""
        f_full = torch.tensor([1.0, 2.0, 3.0])
        f_text = torch.tensor([0.5, 1.0, 1.5])
        f_null = torch.tensor([0.0, 0.0, 0.0])

        result = dcfg_combine(f_full, f_text, f_null, lambda_t=2.0, lambda_a=0.5)

        # 手計算: f_text + 2.0*(f_text - f_null) + 0.5*(f_full - f_text)
        # = [0.5, 1.0, 1.5] + 2.0*[0.5, 1.0, 1.5] + 0.5*[0.5, 1.0, 1.5]
        # = [0.5, 1.0, 1.5] + [1.0, 2.0, 3.0] + [0.25, 0.5, 0.75]
        # = [1.75, 3.5, 5.25]
        expected = torch.tensor([1.75, 3.5, 5.25])
        assert torch.allclose(result, expected)

    def test_zero_lambda_a(self):
        """λ_a=0の場合、参照音声の影響がなくなるか"""
        f_full = torch.tensor([10.0, 20.0, 30.0])  # 参照音声あり
        f_text = torch.tensor([1.0, 2.0, 3.0])  # テキストのみ
        f_null = torch.tensor([0.0, 0.0, 0.0])

        # λ_a=0 の場合、f_full の影響がなくなる
        result = dcfg_combine(f_full, f_text, f_null, lambda_t=2.0, lambda_a=0.0)

        # f_text + 2.0*(f_text - f_null) + 0.0*(f_full - f_text)
        # = f_text + 2.0*f_text = 3.0*f_text
        expected = torch.tensor([3.0, 6.0, 9.0])
        assert torch.allclose(result, expected)

    def test_high_lambda_a(self):
        """λ_a が大きい場合、参照音声の影響が増加するか"""
        f_full = torch.tensor([4.0, 4.0, 4.0])
        f_text = torch.tensor([2.0, 2.0, 2.0])
        f_null = torch.tensor([0.0, 0.0, 0.0])

        # λ_a=0.5 (デフォルト)
        result_low = dcfg_combine(f_full, f_text, f_null, lambda_t=2.0, lambda_a=0.5)

        # λ_a=2.0 (高い)
        result_high = dcfg_combine(f_full, f_text, f_null, lambda_t=2.0, lambda_a=2.0)

        # λ_a が高いほど f_full の影響が大きくなる
        # result_high は f_full により近づく
        assert result_high.mean() > result_low.mean()

    def test_batch_support(self):
        """バッチ処理が正しく動作するか"""
        batch_size = 4
        seq_len = 100
        dim = 256

        f_full = torch.randn(batch_size, seq_len, dim)
        f_text = torch.randn(batch_size, seq_len, dim)
        f_null = torch.randn(batch_size, seq_len, dim)

        result = dcfg_combine(f_full, f_text, f_null)
        assert result.shape == (batch_size, seq_len, dim)

    def test_gradient_flow(self):
        """勾配が正しく流れるか"""
        f_full = torch.randn(2, 10, requires_grad=True)
        f_text = torch.randn(2, 10, requires_grad=True)
        f_null = torch.randn(2, 10, requires_grad=True)

        result = dcfg_combine(f_full, f_text, f_null)
        loss = result.sum()
        loss.backward()

        assert f_full.grad is not None
        assert f_text.grad is not None
        assert f_null.grad is not None

    def test_device_compatibility(self):
        """デバイス互換性テスト（CPU）"""
        f_full = torch.randn(2, 10, device="cpu")
        f_text = torch.randn(2, 10, device="cpu")
        f_null = torch.randn(2, 10, device="cpu")

        result = dcfg_combine(f_full, f_text, f_null)
        assert result.device.type == "cpu"

    def test_dtype_preservation(self):
        """データ型が保持されるか"""
        for dtype in [torch.float32, torch.float64]:
            f_full = torch.randn(2, 10, dtype=dtype)
            f_text = torch.randn(2, 10, dtype=dtype)
            f_null = torch.randn(2, 10, dtype=dtype)

            result = dcfg_combine(f_full, f_text, f_null)
            assert result.dtype == dtype


class TestDCFGFormula:
    """DCFG式の数学的性質のテスト"""

    def test_identity_when_all_same(self):
        """全ての入力が同じ場合、出力も同じになるか"""
        value = torch.randn(4, 10)
        result = dcfg_combine(value, value, value, lambda_t=2.0, lambda_a=0.5)

        # f_text + λ_t*(f_text - f_null) + λ_a*(f_full - f_text)
        # = value + 2.0*(value - value) + 0.5*(value - value)
        # = value
        assert torch.allclose(result, value)

    def test_linearity_in_lambda_t(self):
        """λ_t に対する線形性"""
        f_full = torch.randn(2, 10)
        f_text = torch.randn(2, 10)
        f_null = torch.randn(2, 10)

        # λ_t=0 の場合
        result_0 = dcfg_combine(f_full, f_text, f_null, lambda_t=0.0, lambda_a=0.5)

        # λ_t=4 の場合
        result_4 = dcfg_combine(f_full, f_text, f_null, lambda_t=4.0, lambda_a=0.5)

        # λ_t=2 の場合（中間）
        result_2 = dcfg_combine(f_full, f_text, f_null, lambda_t=2.0, lambda_a=0.5)

        # result_2 は result_0 と result_4 の中間に近いはず
        # （厳密には λ_t に関して線形なので）
        diff_text = f_text - f_null
        expected_diff = 2.0 * diff_text
        actual_diff = result_2 - result_0
        assert torch.allclose(actual_diff, expected_diff)


class TestDCFGIntegration:
    """DCFG統合テスト"""

    def test_import_from_package(self):
        """パッケージからインポートできるか"""
        from f5_tts.restyle import DCFGConfig, dcfg_combine

        assert DCFGConfig is not None
        assert dcfg_combine is not None

    def test_cfm_has_dcfg_params(self):
        """CFMクラスにDCFGパラメータがあるか"""
        import inspect

        from f5_tts.model.cfm import CFM

        # sample メソッドのシグネチャを確認
        sig = inspect.signature(CFM.sample)
        params = list(sig.parameters.keys())

        assert "use_dcfg" in params
        assert "lambda_t" in params
        assert "lambda_a" in params

    def test_dit_has_dcfg_infer(self):
        """DiTクラスにdcfg_inferパラメータがあるか"""
        import inspect

        from f5_tts.model.backbones.dit import DiT

        # forward メソッドのシグネチャを確認
        sig = inspect.signature(DiT.forward)
        params = list(sig.parameters.keys())

        assert "dcfg_infer" in params


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestDCFGCUDA:
    """CUDA環境でのテスト"""

    def test_cuda_device(self):
        """CUDAデバイスで動作するか"""
        f_full = torch.randn(2, 10, device="cuda")
        f_text = torch.randn(2, 10, device="cuda")
        f_null = torch.randn(2, 10, device="cuda")

        result = dcfg_combine(f_full, f_text, f_null)
        assert result.device.type == "cuda"

    def test_cuda_half_precision(self):
        """半精度（float16）で動作するか"""
        f_full = torch.randn(2, 10, device="cuda", dtype=torch.float16)
        f_text = torch.randn(2, 10, device="cuda", dtype=torch.float16)
        f_null = torch.randn(2, 10, device="cuda", dtype=torch.float16)

        result = dcfg_combine(f_full, f_text, f_null)
        assert result.dtype == torch.float16
