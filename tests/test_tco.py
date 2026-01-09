"""
TCO (Timbre Consistency Optimization) のテスト

ReStyle-TTS Phase 4 の実装テスト
"""

import pytest
import torch
import torch.nn as nn

from f5_tts.restyle.tco import (
    TCOConfig,
    TCOWeightComputer,
    TCOLoss,
    TCOTrainingMixin,
    create_tco_loss,
)
from f5_tts.restyle.speaker_encoder import (
    SpeakerEncoderConfig,
    SpeakerEncoder,
    compute_speaker_similarity,
)


class TestTCOConfig:
    """TCOConfig のテスト"""

    def test_default_values(self):
        """デフォルト値が正しいか"""
        config = TCOConfig()
        assert config.lambda_reward == 0.2
        assert config.beta == 5.0
        assert config.mu == 0.9
        assert config.min_weight == 0.1
        assert config.max_weight == 2.0
        assert config.enabled is True

    def test_custom_values(self):
        """カスタム値が設定できるか"""
        config = TCOConfig(
            lambda_reward=0.5,
            beta=10.0,
            mu=0.95,
            enabled=False,
        )
        assert config.lambda_reward == 0.5
        assert config.beta == 10.0
        assert config.mu == 0.95
        assert config.enabled is False


class TestTCOWeightComputer:
    """TCOWeightComputer のテスト"""

    def test_initialization(self):
        """初期化"""
        computer = TCOWeightComputer()
        assert computer.config is not None
        assert computer.baseline.item() == 0.5
        assert computer.num_updates.item() == 0

    def test_custom_config(self):
        """カスタム設定"""
        config = TCOConfig(lambda_reward=0.5)
        computer = TCOWeightComputer(config)
        assert computer.config.lambda_reward == 0.5

    def test_compute_advantage(self):
        """アドバンテージ計算"""
        computer = TCOWeightComputer()
        reward = torch.tensor([0.8, 0.6, 0.4])

        advantage = computer.compute_advantage(reward)

        # baseline = 0.5 なので advantage = reward - 0.5
        expected = torch.tensor([0.3, 0.1, -0.1])
        assert torch.allclose(advantage, expected)

    def test_compute_weight(self):
        """重み計算"""
        config = TCOConfig(lambda_reward=0.2, beta=5.0)
        computer = TCOWeightComputer(config)

        # 正のアドバンテージ -> 重み > 1
        pos_adv = torch.tensor([0.5])
        weight_pos = computer.compute_weight(pos_adv)
        assert weight_pos.item() > 1.0

        # 負のアドバンテージ -> 重み < 1
        neg_adv = torch.tensor([-0.5])
        weight_neg = computer.compute_weight(neg_adv)
        assert weight_neg.item() < 1.0

        # ゼロアドバンテージ -> 重み = 1
        zero_adv = torch.tensor([0.0])
        weight_zero = computer.compute_weight(zero_adv)
        assert abs(weight_zero.item() - 1.0) < 1e-5

    def test_weight_clipping(self):
        """重みクリッピング"""
        config = TCOConfig(min_weight=0.5, max_weight=1.5)
        computer = TCOWeightComputer(config)

        # 極端な正のアドバンテージ
        large_adv = torch.tensor([10.0])
        weight = computer.compute_weight(large_adv)
        assert weight.item() <= 1.5

        # 極端な負のアドバンテージ
        small_adv = torch.tensor([-10.0])
        weight = computer.compute_weight(small_adv)
        assert weight.item() >= 0.5

    def test_update_baseline_first(self):
        """初回のベースライン更新"""
        computer = TCOWeightComputer()
        reward = torch.tensor([0.8])

        computer.update_baseline(reward)

        assert computer.num_updates.item() == 1
        assert abs(computer.baseline.item() - 0.8) < 1e-5

    def test_update_baseline_ema(self):
        """EMAベースライン更新"""
        config = TCOConfig(mu=0.9)
        computer = TCOWeightComputer(config)

        # 初回更新
        computer.update_baseline(torch.tensor([0.5]))
        baseline1 = computer.baseline.item()

        # 2回目更新
        computer.update_baseline(torch.tensor([1.0]))
        baseline2 = computer.baseline.item()

        # EMA: 0.9 * 0.5 + 0.1 * 1.0 = 0.55
        expected = 0.9 * 0.5 + 0.1 * 1.0
        assert abs(baseline2 - expected) < 1e-5

    def test_forward(self):
        """forward メソッド"""
        computer = TCOWeightComputer()
        computer.train()

        reward = torch.tensor([0.8, 0.6, 0.4])
        weight = computer(reward)

        assert weight.shape == reward.shape
        assert (weight >= computer.config.min_weight).all()
        assert (weight <= computer.config.max_weight).all()

    def test_forward_disabled(self):
        """無効時は全て1"""
        config = TCOConfig(enabled=False)
        computer = TCOWeightComputer(config)

        reward = torch.tensor([0.8, 0.6, 0.4])
        weight = computer(reward)

        assert torch.allclose(weight, torch.ones_like(reward))


class TestTCOLoss:
    """TCOLoss のテスト"""

    def test_initialization(self):
        """初期化"""
        loss = TCOLoss()
        assert loss.config is not None
        assert loss.weight_computer is not None

    def test_forward_with_reward(self):
        """事前計算報酬での forward"""
        loss = TCOLoss()
        loss.train()

        base_loss = torch.tensor(1.0)
        reward = torch.tensor([0.8, 0.6])

        weighted_loss, metrics = loss(base_loss, reward=reward)

        assert weighted_loss.shape == base_loss.shape
        assert "tco_enabled" in metrics
        assert metrics["tco_enabled"] is True
        assert "tco_reward_mean" in metrics
        assert "tco_weight_mean" in metrics

    def test_forward_disabled(self):
        """無効時"""
        config = TCOConfig(enabled=False)
        loss = TCOLoss(config=config)

        base_loss = torch.tensor(1.0)
        reward = torch.tensor([0.8])

        weighted_loss, metrics = loss(base_loss, reward=reward)

        assert torch.allclose(weighted_loss, base_loss)
        assert metrics["tco_enabled"] is False

    def test_forward_batch_loss(self):
        """バッチ損失"""
        loss = TCOLoss()
        loss.train()

        base_loss = torch.tensor([1.0, 2.0, 3.0])
        reward = torch.tensor([0.8, 0.6, 0.4])

        weighted_loss, metrics = loss(base_loss, reward=reward)

        # 重み付き平均
        assert weighted_loss.dim() == 0

    def test_forward_requires_audio_or_reward(self):
        """音声またはrewardが必要"""
        loss = TCOLoss()

        base_loss = torch.tensor(1.0)

        with pytest.raises(ValueError, match="reward が指定されていない"):
            loss(base_loss)


class TestSpeakerEncoderConfig:
    """SpeakerEncoderConfig のテスト"""

    def test_default_values(self):
        """デフォルト値が正しいか"""
        config = SpeakerEncoderConfig()
        assert config.model_name == "microsoft/wavlm-base-plus-sv"
        assert config.pooling == "mean"
        assert config.normalize is True

    def test_custom_values(self):
        """カスタム値が設定できるか"""
        config = SpeakerEncoderConfig(
            model_name="custom-model",
            pooling="first",
            normalize=False,
        )
        assert config.model_name == "custom-model"
        assert config.pooling == "first"
        assert config.normalize is False


class TestSpeakerEncoder:
    """SpeakerEncoder のテスト（モデル読み込みなし）"""

    def test_initialization(self):
        """初期化（モデル読み込みなし）"""
        encoder = SpeakerEncoder()
        assert encoder.config is not None
        assert encoder._model is None  # 遅延読み込み

    def test_custom_config(self):
        """カスタム設定"""
        config = SpeakerEncoderConfig(pooling="first")
        encoder = SpeakerEncoder(config)
        assert encoder.config.pooling == "first"


class TestSpeakerEncoderWithModel:
    """SpeakerEncoder のテスト（モデル読み込みあり）"""

    @pytest.fixture
    def encoder(self):
        """エンコーダー（遅延読み込み）"""
        return SpeakerEncoder()

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_cuda_device(self, encoder):
        """CUDAデバイス"""
        config = SpeakerEncoderConfig(device="cuda")
        cuda_encoder = SpeakerEncoder(config)
        # モデル読み込みをトリガー
        _ = cuda_encoder.device
        assert cuda_encoder.device.type == "cuda"

    def test_compute_similarity_identical(self):
        """同一埋め込みの類似度は1"""
        encoder = SpeakerEncoder()
        emb = torch.randn(10)
        emb = torch.nn.functional.normalize(emb, dim=-1)

        sim = encoder.compute_similarity(emb, emb)
        assert abs(sim.item() - 1.0) < 1e-5

    def test_compute_similarity_orthogonal(self):
        """直交埋め込みの類似度は0"""
        encoder = SpeakerEncoder()
        emb1 = torch.tensor([1.0, 0.0, 0.0])
        emb2 = torch.tensor([0.0, 1.0, 0.0])

        sim = encoder.compute_similarity(emb1, emb2)
        assert abs(sim.item()) < 1e-5

    def test_compute_similarity_batch(self):
        """バッチ類似度"""
        encoder = SpeakerEncoder()
        emb1 = torch.randn(4, 10)
        emb2 = torch.randn(4, 10)

        sim = encoder.compute_similarity(emb1, emb2)
        assert sim.shape == torch.Size([4])


class TestTCOTrainingMixin:
    """TCOTrainingMixin のテスト"""

    def test_init_tco(self):
        """TCO初期化"""

        class DummyTrainer(TCOTrainingMixin):
            pass

        trainer = DummyTrainer()
        trainer.init_tco()

        assert hasattr(trainer, "tco_config")
        assert hasattr(trainer, "tco_loss")

    def test_apply_tco_weight(self):
        """TCO重み適用"""

        class DummyTrainer(TCOTrainingMixin):
            pass

        trainer = DummyTrainer()
        trainer.init_tco()
        trainer.tco_loss.train()

        loss = torch.tensor(1.0)
        # 事前計算報酬を使用
        weighted_loss, metrics = trainer.tco_loss(loss, reward=torch.tensor([0.8]))

        assert weighted_loss is not None
        assert "tco_enabled" in metrics


class TestCreateTCOLoss:
    """create_tco_loss のテスト"""

    def test_create_default(self):
        """デフォルト設定で作成"""
        loss = create_tco_loss()
        assert isinstance(loss, TCOLoss)
        assert loss.config.enabled is True

    def test_create_with_config(self):
        """カスタム設定で作成"""
        config = TCOConfig(lambda_reward=0.5)
        loss = create_tco_loss(config)
        assert loss.config.lambda_reward == 0.5


class TestIntegration:
    """統合テスト"""

    def test_import_from_package(self):
        """パッケージからインポートできるか"""
        from f5_tts.restyle import (
            TCOConfig,
            TCOWeightComputer,
            TCOLoss,
            TCOTrainingMixin,
            create_tco_loss,
            SpeakerEncoderConfig,
            SpeakerEncoder,
            compute_speaker_similarity,
        )

        assert TCOConfig is not None
        assert TCOWeightComputer is not None
        assert TCOLoss is not None
        assert SpeakerEncoder is not None

    def test_weight_computation_formula(self):
        """重み計算式の検証: w = 1 + λ * tanh(β * A)"""
        config = TCOConfig(lambda_reward=0.2, beta=5.0, mu=0.9)
        computer = TCOWeightComputer(config)

        # 固定ベースライン
        computer.baseline = torch.tensor(0.5)

        reward = torch.tensor([0.7])  # A = 0.7 - 0.5 = 0.2
        weight = computer(reward, update_baseline=False)

        # 手計算: w = 1 + 0.2 * tanh(5.0 * 0.2) = 1 + 0.2 * tanh(1.0)
        expected = 1.0 + 0.2 * torch.tanh(torch.tensor(1.0))
        assert torch.allclose(weight, expected.unsqueeze(0), atol=1e-5)

    def test_full_tco_pipeline(self):
        """完全なTCOパイプライン"""
        # TCO設定
        config = TCOConfig(
            lambda_reward=0.2,
            beta=5.0,
            mu=0.9,
        )

        # TCOLoss作成
        tco_loss = TCOLoss(config=config)
        tco_loss.train()

        # ダミーの損失と報酬
        base_loss = torch.tensor([1.0, 1.5, 2.0])
        reward = torch.tensor([0.9, 0.5, 0.3])

        # TCO重み付き損失
        weighted_loss, metrics = tco_loss(base_loss, reward=reward)

        # 検証
        assert weighted_loss.dim() == 0  # スカラー
        assert metrics["tco_enabled"] is True
        assert 0.3 <= metrics["tco_reward_mean"] <= 0.9
        assert 0.1 <= metrics["tco_weight_mean"] <= 2.0

        # 2回目の呼び出し（ベースライン更新確認）
        _, metrics2 = tco_loss(base_loss, reward=reward)
        # ベースラインが更新されているはず
        assert metrics2["tco_baseline"] != 0.5
