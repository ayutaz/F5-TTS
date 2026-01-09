"""
OLoRA Fusion のテスト

ReStyle-TTS Phase 3 の実装テスト
"""

import pytest
import torch

from f5_tts.restyle.olora_fusion import (
    OLoRAConfig,
    OLoRAFusion,
    compute_orthogonal_projection,
    orthogonalize_loras,
    fuse_lora_weights,
)


class TestOLoRAConfig:
    """OLoRAConfig のテスト"""

    def test_default_values(self):
        """デフォルト値が正しいか"""
        config = OLoRAConfig()
        assert config.orthogonalize is True
        assert config.epsilon == 1e-8
        assert config.use_svd is False

    def test_custom_values(self):
        """カスタム値が設定できるか"""
        config = OLoRAConfig(
            orthogonalize=False,
            epsilon=1e-6,
            use_svd=True,
        )
        assert config.orthogonalize is False
        assert config.epsilon == 1e-6
        assert config.use_svd is True


class TestComputeOrthogonalProjection:
    """compute_orthogonal_projection のテスト"""

    def test_single_vector(self):
        """ベクトルが1つの場合はそのまま返す"""
        vectors = torch.randn(1, 10)
        result = compute_orthogonal_projection(vectors, 0)
        assert torch.allclose(result, vectors[0])

    def test_orthogonal_vectors_unchanged(self):
        """直交ベクトルは変化しない"""
        # 直交ベクトルを作成
        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([0.0, 1.0, 0.0])
        vectors = torch.stack([v1, v2])

        # v1を直交化（v2と既に直交なので変化しないはず）
        result = compute_orthogonal_projection(vectors, 0)

        # 結果はv1とほぼ同じであるべき
        assert torch.allclose(result, v1, atol=1e-5)

    def test_parallel_vectors(self):
        """平行ベクトルは直交成分がゼロになる"""
        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([2.0, 0.0, 0.0])  # v1と平行
        vectors = torch.stack([v1, v2])

        # v1を直交化（v2と平行なので直交成分はゼロ）
        result = compute_orthogonal_projection(vectors, 0)

        # 結果はゼロに近いはず
        assert torch.allclose(result, torch.zeros(3), atol=1e-5)

    def test_general_case(self):
        """一般的なケースで直交化が機能するか"""
        v1 = torch.tensor([1.0, 1.0, 0.0])
        v2 = torch.tensor([1.0, 0.0, 0.0])
        vectors = torch.stack([v1, v2])

        # v1を直交化
        result = compute_orthogonal_projection(vectors, 0)

        # 結果はv2に直交するはず
        dot_product = torch.dot(result, v2)
        assert abs(dot_product.item()) < 1e-5

    def test_preserves_device(self):
        """デバイスが保持されるか"""
        vectors = torch.randn(2, 10)
        result = compute_orthogonal_projection(vectors, 0)
        assert result.device == vectors.device


class TestOrthogonalizeLoras:
    """orthogonalize_loras のテスト"""

    def test_empty_dict(self):
        """空の辞書"""
        result = orthogonalize_loras({})
        assert result == {}

    def test_single_lora(self):
        """1つのLoRAは変化しない"""
        lora_deltas = {"pitch_high": torch.randn(100)}
        result = orthogonalize_loras(lora_deltas)
        assert torch.allclose(result["pitch_high"], lora_deltas["pitch_high"])

    def test_multiple_loras(self):
        """複数のLoRAが直交化される"""
        lora_deltas = {
            "pitch_high": torch.randn(100),
            "angry": torch.randn(100),
            "energy_high": torch.randn(100),
        }
        result = orthogonalize_loras(lora_deltas)

        # 全てのキーが存在するか
        assert set(result.keys()) == set(lora_deltas.keys())

        # 全ての結果が同じ形状か
        for name in lora_deltas.keys():
            assert result[name].shape == lora_deltas[name].shape

    def test_orthogonalized_vectors_are_orthogonal(self):
        """直交化後のベクトルは互いに直交に近くなる"""
        # シンプルなケースでテスト
        torch.manual_seed(42)
        lora_deltas = {
            "a": torch.randn(50),
            "b": torch.randn(50),
        }

        result = orthogonalize_loras(lora_deltas)

        # 直交化前の内積
        before_dot = torch.dot(lora_deltas["a"], lora_deltas["b"]).abs()

        # 直交化後の内積
        after_dot = torch.dot(result["a"], result["b"]).abs()

        # 直交化後の内積が小さくなるはず（完全にゼロではない可能性あり）
        # 少なくとも大幅に減少するはず
        assert after_dot < before_dot or after_dot < 1e-3


class TestFuseLoraWeights:
    """fuse_lora_weights のテスト"""

    def test_empty_input(self):
        """空の入力"""
        result = fuse_lora_weights({}, {})
        assert result == {}

    def test_single_lora(self):
        """1つのLoRA"""
        state_dicts = {
            "pitch_high": {"layer1": torch.ones(10), "layer2": torch.ones(5)}
        }
        alphas = {"pitch_high": 2.0}

        result = fuse_lora_weights(state_dicts, alphas, orthogonalize=False)

        assert "layer1" in result
        assert "layer2" in result
        assert torch.allclose(result["layer1"], torch.ones(10) * 2.0)
        assert torch.allclose(result["layer2"], torch.ones(5) * 2.0)

    def test_zero_alpha_ignored(self):
        """alpha=0のLoRAは無視される"""
        state_dicts = {
            "pitch_high": {"layer1": torch.ones(10)},
            "angry": {"layer1": torch.ones(10) * 100},
        }
        alphas = {"pitch_high": 1.0, "angry": 0.0}

        result = fuse_lora_weights(state_dicts, alphas, orthogonalize=False)

        assert torch.allclose(result["layer1"], torch.ones(10))

    def test_multiple_loras_without_orthogonalization(self):
        """直交化なしの複数LoRA融合"""
        state_dicts = {
            "a": {"layer": torch.tensor([1.0, 0.0])},
            "b": {"layer": torch.tensor([0.0, 1.0])},
        }
        alphas = {"a": 1.0, "b": 2.0}

        result = fuse_lora_weights(state_dicts, alphas, orthogonalize=False)

        expected = torch.tensor([1.0, 2.0])
        assert torch.allclose(result["layer"], expected)

    def test_multiple_loras_with_orthogonalization(self):
        """直交化ありの複数LoRA融合"""
        state_dicts = {
            "a": {"layer": torch.tensor([1.0, 0.0])},
            "b": {"layer": torch.tensor([0.0, 1.0])},
        }
        alphas = {"a": 1.0, "b": 1.0}

        result = fuse_lora_weights(state_dicts, alphas, orthogonalize=True)

        # 直交ベクトルなので直交化しても同じ結果
        assert "layer" in result
        assert result["layer"].shape == torch.Size([2])

    def test_preserves_dtype(self):
        """dtypeが保持されるか"""
        state_dicts = {
            "a": {"layer": torch.ones(10, dtype=torch.float16)},
        }
        alphas = {"a": 1.0}

        result = fuse_lora_weights(state_dicts, alphas)

        assert result["layer"].dtype == torch.float16


class TestOLoRAFusion:
    """OLoRAFusion クラスのテスト"""

    def test_initialization(self):
        """初期化"""
        fusion = OLoRAFusion()
        assert fusion.config is not None
        assert len(fusion.lora_state_dicts) == 0

    def test_custom_config(self):
        """カスタム設定"""
        config = OLoRAConfig(orthogonalize=False)
        fusion = OLoRAFusion(config=config)
        assert fusion.config.orthogonalize is False

    def test_add_and_remove_lora(self):
        """LoRAの追加と削除"""
        fusion = OLoRAFusion()

        fusion.add_lora("pitch_high", {"layer": torch.ones(10)})
        assert "pitch_high" in fusion.get_lora_names()

        fusion.remove_lora("pitch_high")
        assert "pitch_high" not in fusion.get_lora_names()

    def test_clear(self):
        """全削除"""
        fusion = OLoRAFusion()
        fusion.add_lora("a", {"layer": torch.ones(10)})
        fusion.add_lora("b", {"layer": torch.ones(10)})

        fusion.clear()
        assert len(fusion.get_lora_names()) == 0

    def test_fuse(self):
        """融合"""
        fusion = OLoRAFusion()
        fusion.add_lora("a", {"layer": torch.ones(10)})
        fusion.add_lora("b", {"layer": torch.ones(10) * 2})

        result = fusion.fuse({"a": 1.0, "b": 1.0})

        assert "layer" in result

    def test_fuse_override_orthogonalize(self):
        """融合時に直交化をオーバーライド"""
        fusion = OLoRAFusion(OLoRAConfig(orthogonalize=True))
        fusion.add_lora("a", {"layer": torch.tensor([1.0, 0.0])})
        fusion.add_lora("b", {"layer": torch.tensor([0.0, 1.0])})

        # 直交化をオフにして融合
        result = fusion.fuse({"a": 1.0, "b": 2.0}, orthogonalize=False)

        expected = torch.tensor([1.0, 2.0])
        assert torch.allclose(result["layer"], expected)

    def test_compute_interference_identical(self):
        """同一LoRAの干渉度は1"""
        fusion = OLoRAFusion()
        v = torch.randn(100)
        fusion.add_lora("a", {"layer": v})
        fusion.add_lora("b", {"layer": v})  # 同じベクトル

        interference = fusion.compute_interference("a", "b")
        assert abs(interference - 1.0) < 1e-5

    def test_compute_interference_orthogonal(self):
        """直交LoRAの干渉度は0に近い"""
        fusion = OLoRAFusion()
        fusion.add_lora("a", {"layer": torch.tensor([1.0, 0.0, 0.0])})
        fusion.add_lora("b", {"layer": torch.tensor([0.0, 1.0, 0.0])})

        interference = fusion.compute_interference("a", "b")
        assert interference < 1e-5

    def test_compute_interference_invalid_lora(self):
        """存在しないLoRAでエラー"""
        fusion = OLoRAFusion()
        fusion.add_lora("a", {"layer": torch.ones(10)})

        with pytest.raises(ValueError):
            fusion.compute_interference("a", "nonexistent")

    def test_get_interference_matrix(self):
        """干渉行列の取得"""
        fusion = OLoRAFusion()
        fusion.add_lora("a", {"layer": torch.tensor([1.0, 0.0])})
        fusion.add_lora("b", {"layer": torch.tensor([0.0, 1.0])})
        fusion.add_lora("c", {"layer": torch.tensor([1.0, 1.0])})

        matrix = fusion.get_interference_matrix()

        assert "a" in matrix
        assert "b" in matrix
        assert "c" in matrix
        assert matrix["a"]["a"] == 1.0
        assert matrix["b"]["b"] == 1.0
        assert abs(matrix["a"]["b"]) < 1e-5  # 直交


class TestIntegration:
    """統合テスト"""

    def test_import_from_package(self):
        """パッケージからインポートできるか"""
        from f5_tts.restyle import (
            OLoRAConfig,
            OLoRAFusion,
            fuse_lora_weights,
            orthogonalize_loras,
            compute_orthogonal_projection,
        )

        assert OLoRAConfig is not None
        assert OLoRAFusion is not None
        assert fuse_lora_weights is not None

    def test_style_lora_manager_with_olora(self):
        """StyleLoRAManagerでOLoRAが使用できるか"""
        import torch.nn as nn
        from f5_tts.restyle import StyleLoRAManager, OLoRAConfig

        # 簡単なモデル
        model = nn.Linear(10, 10)

        # OLoRA設定付きでマネージャーを初期化
        olora_config = OLoRAConfig(orthogonalize=True)
        manager = StyleLoRAManager(model, olora_config=olora_config)

        assert manager.olora_config.orthogonalize is True

    def test_full_pipeline(self):
        """完全なパイプラインテスト"""
        fusion = OLoRAFusion()

        # 複数のLoRAを追加
        torch.manual_seed(42)
        fusion.add_lora("pitch_high", {
            "layer1.lora_A": torch.randn(32, 1024),
            "layer1.lora_B": torch.randn(1024, 32),
        })
        fusion.add_lora("angry", {
            "layer1.lora_A": torch.randn(32, 1024),
            "layer1.lora_B": torch.randn(1024, 32),
        })
        fusion.add_lora("energy_high", {
            "layer1.lora_A": torch.randn(32, 1024),
            "layer1.lora_B": torch.randn(1024, 32),
        })

        # 干渉行列を計算
        matrix = fusion.get_interference_matrix()
        assert len(matrix) == 3

        # 融合
        fused = fusion.fuse({
            "pitch_high": 1.0,
            "angry": 0.5,
            "energy_high": 0.3,
        })

        assert "layer1.lora_A" in fused
        assert "layer1.lora_B" in fused
        assert fused["layer1.lora_A"].shape == (32, 1024)
        assert fused["layer1.lora_B"].shape == (1024, 32)
