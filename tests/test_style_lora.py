"""
Style LoRA のテスト

ReStyle-TTS Phase 2 の実装テスト
"""

import pytest
import torch
import torch.nn as nn

from f5_tts.restyle.style_lora import (
    STYLE_ATTRIBUTES,
    STYLE_CATEGORIES,
    StyleLoRAConfig,
    StyleLoRAManager,
    create_lora_model,
    freeze_base_model,
    unfreeze_lora_params,
    count_trainable_params,
    DEFAULT_TARGET_MODULES,
)


class TestStyleAttributes:
    """スタイル属性定義のテスト"""

    def test_style_attributes_defined(self):
        """スタイル属性が定義されているか"""
        assert len(STYLE_ATTRIBUTES) == 10

        # ピッチ
        assert "pitch_high" in STYLE_ATTRIBUTES
        assert "pitch_low" in STYLE_ATTRIBUTES

        # エネルギー
        assert "energy_high" in STYLE_ATTRIBUTES
        assert "energy_low" in STYLE_ATTRIBUTES

        # 感情
        assert "angry" in STYLE_ATTRIBUTES
        assert "happy" in STYLE_ATTRIBUTES
        assert "sad" in STYLE_ATTRIBUTES
        assert "fear" in STYLE_ATTRIBUTES
        assert "disgusted" in STYLE_ATTRIBUTES
        assert "surprised" in STYLE_ATTRIBUTES

    def test_style_categories(self):
        """スタイルカテゴリが正しいか"""
        assert "pitch" in STYLE_CATEGORIES
        assert "energy" in STYLE_CATEGORIES
        assert "emotion" in STYLE_CATEGORIES

        assert len(STYLE_CATEGORIES["pitch"]) == 2
        assert len(STYLE_CATEGORIES["energy"]) == 2
        assert len(STYLE_CATEGORIES["emotion"]) == 6

    def test_all_attributes_in_categories(self):
        """全属性がカテゴリに含まれているか"""
        all_in_categories = []
        for styles in STYLE_CATEGORIES.values():
            all_in_categories.extend(styles)

        for attr in STYLE_ATTRIBUTES:
            assert attr in all_in_categories


class TestStyleLoRAConfig:
    """StyleLoRAConfig のテスト"""

    def test_default_values(self):
        """デフォルト値が正しいか"""
        config = StyleLoRAConfig()
        assert config.rank == 32
        assert config.alpha == 64
        assert config.dropout == 0.0
        assert config.bias == "none"
        assert config.target_modules == DEFAULT_TARGET_MODULES

    def test_custom_values(self):
        """カスタム値が設定できるか"""
        config = StyleLoRAConfig(
            rank=16,
            alpha=32,
            dropout=0.1,
            target_modules=["to_q", "to_k"],
        )
        assert config.rank == 16
        assert config.alpha == 32
        assert config.dropout == 0.1
        assert config.target_modules == ["to_q", "to_k"]

    def test_to_lora_config(self):
        """peftのLoraConfigに変換できるか"""
        config = StyleLoRAConfig(rank=16, alpha=32)
        lora_config = config.to_lora_config()

        assert lora_config.r == 16
        assert lora_config.lora_alpha == 32
        assert lora_config.lora_dropout == 0.0


class TestStyleLoRAManager:
    """StyleLoRAManager のテスト"""

    @pytest.fixture
    def simple_model(self):
        """テスト用の簡単なモデル"""
        return nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def test_initialization(self, simple_model):
        """初期化が正しいか"""
        manager = StyleLoRAManager(simple_model)
        assert manager.base_model is simple_model
        assert manager.config is not None
        assert len(manager.loaded_loras) == 0

    def test_custom_config(self, simple_model):
        """カスタム設定で初期化できるか"""
        config = StyleLoRAConfig(rank=16)
        manager = StyleLoRAManager(simple_model, config=config)
        assert manager.config.rank == 16

    def test_get_loaded_styles_empty(self, simple_model):
        """読み込みスタイルが空の場合"""
        manager = StyleLoRAManager(simple_model)
        assert manager.get_loaded_styles() == []

    def test_unload_all(self, simple_model):
        """全アンロードが動作するか"""
        manager = StyleLoRAManager(simple_model)
        # 手動でダミーデータを追加
        manager.loaded_loras["test"] = {"dummy": torch.zeros(1)}
        manager.unload_all()
        assert len(manager.loaded_loras) == 0

    def test_get_style_info(self, simple_model):
        """スタイル情報を取得できるか"""
        manager = StyleLoRAManager(simple_model)
        info = manager.get_style_info("pitch_high")

        assert info["name"] == "pitch_high"
        assert info["description"] == "高いピッチ"
        assert info["category"] == "pitch"
        assert info["loaded"] is False

    def test_get_style_info_invalid(self, simple_model):
        """不正なスタイル名でエラーになるか"""
        manager = StyleLoRAManager(simple_model)
        with pytest.raises(ValueError, match="不明なスタイル属性"):
            manager.get_style_info("invalid_style")

    def test_load_lora_invalid_style(self, simple_model, tmp_path):
        """不正なスタイル名でエラーになるか"""
        manager = StyleLoRAManager(simple_model)
        dummy_path = tmp_path / "dummy.pt"
        dummy_path.touch()

        with pytest.raises(ValueError, match="不明なスタイル属性"):
            manager.load_lora("invalid_style", dummy_path)

    def test_load_lora_file_not_found(self, simple_model):
        """存在しないファイルでエラーになるか"""
        manager = StyleLoRAManager(simple_model)
        with pytest.raises(FileNotFoundError):
            manager.load_lora("pitch_high", "/nonexistent/path.pt")


class TestFreezeUnfreeze:
    """凍結・解凍機能のテスト"""

    @pytest.fixture
    def model(self):
        """テスト用モデル"""
        return nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

    def test_freeze_base_model(self, model):
        """ベースモデルを凍結できるか"""
        freeze_base_model(model)

        for param in model.parameters():
            assert param.requires_grad is False

    def test_count_trainable_params_frozen(self, model):
        """凍結後のパラメータ数が正しいか"""
        freeze_base_model(model)
        trainable, total = count_trainable_params(model)

        assert trainable == 0
        assert total > 0

    def test_count_trainable_params_unfrozen(self, model):
        """凍結前のパラメータ数が正しいか"""
        trainable, total = count_trainable_params(model)
        assert trainable == total
        assert total > 0


class TestCreateLoRAModel:
    """create_lora_model のテスト"""

    @pytest.fixture
    def model_with_targets(self):
        """LoRAターゲットを持つモデル"""

        class SimpleTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = nn.Linear(64, 64)
                self.to_k = nn.Linear(64, 64)
                self.to_v = nn.Linear(64, 64)
                self.out = nn.Linear(64, 64)

            def forward(self, x):
                q = self.to_q(x)
                k = self.to_k(x)
                v = self.to_v(x)
                return self.out(q + k + v)

        return SimpleTransformer()

    def test_create_lora_model(self, model_with_targets):
        """LoRAモデルを作成できるか"""
        config = StyleLoRAConfig(
            rank=8,
            target_modules=["to_q", "to_k", "to_v"],
        )
        peft_model = create_lora_model(model_with_targets, config)

        # PeftModelが返されることを確認
        assert hasattr(peft_model, "get_base_model")

    def test_lora_params_added(self, model_with_targets):
        """LoRAパラメータが追加されているか"""
        config = StyleLoRAConfig(
            rank=8,
            target_modules=["to_q", "to_k", "to_v"],
        )
        peft_model = create_lora_model(model_with_targets, config)

        # LoRAパラメータを探す
        lora_params = [
            name for name, _ in peft_model.named_parameters() if "lora_" in name
        ]
        assert len(lora_params) > 0


class TestIntegration:
    """統合テスト"""

    def test_import_from_package(self):
        """パッケージからインポートできるか"""
        from f5_tts.restyle import (
            STYLE_ATTRIBUTES,
            STYLE_CATEGORIES,
            StyleLoRAConfig,
            StyleLoRAManager,
            create_lora_model,
            freeze_base_model,
            unfreeze_lora_params,
            count_trainable_params,
        )

        assert STYLE_ATTRIBUTES is not None
        assert STYLE_CATEGORIES is not None
        assert StyleLoRAConfig is not None
        assert StyleLoRAManager is not None

    def test_config_file_exists(self):
        """設定ファイルが存在するか"""
        from importlib.resources import files

        config_path = files("f5_tts").joinpath("configs/ReStyleTTS_Base.yaml")
        assert config_path.is_file()
