"""
Style LoRA training script for ReStyle-TTS

ベースモデルを凍結し、スタイル属性に特化したLoRAアダプターを訓練する。

Usage:
    python -m f5_tts.train.train_style_lora --config-name ReStyleTTS_Base \
        style_attribute=pitch_high \
        datasets.name=your_dataset

Reference: ReStyle-TTS (arXiv:2601.03632) Section 3.2
"""

import os
from importlib.resources import files
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from peft import get_peft_model

from f5_tts.model import CFM, Trainer
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer
from f5_tts.restyle.style_lora import (
    STYLE_ATTRIBUTES,
    StyleLoRAConfig,
    freeze_base_model,
    unfreeze_lora_params,
    count_trainable_params,
)


os.chdir(str(files("f5_tts").joinpath("../..")))


def load_base_model(model_cfg: DictConfig) -> CFM:
    """ベースモデルを読み込み

    Args:
        model_cfg: モデル設定

    Returns:
        CFMモデル
    """
    model_cls = hydra.utils.get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    tokenizer = model_cfg.model.tokenizer

    # トークナイザー設定
    if tokenizer != "custom":
        tokenizer_path = model_cfg.datasets.name
    else:
        tokenizer_path = model_cfg.model.tokenizer_path
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    # モデル作成
    model = CFM(
        transformer=model_cls(
            **model_arc,
            text_num_embeds=vocab_size,
            mel_dim=model_cfg.model.mel_spec.n_mel_channels,
        ),
        mel_spec_kwargs=model_cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )

    return model


def load_pretrained_weights(model: CFM, checkpoint_path: str) -> None:
    """事前学習済み重みを読み込み

    Args:
        model: CFMモデル
        checkpoint_path: チェックポイントパス
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"チェックポイントが見つかりません: {checkpoint_path}")

    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # EMAモデルの場合はema_model_state_dictを使用
    if "ema_model_state_dict" in state_dict:
        state_dict = state_dict["ema_model_state_dict"]
    elif "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict, strict=False)
    print(f"事前学習済み重みを読み込みました: {checkpoint_path}")


def setup_lora_model(model: CFM, lora_cfg: DictConfig) -> CFM:
    """LoRAアダプターを追加

    Args:
        model: CFMモデル
        lora_cfg: LoRA設定

    Returns:
        LoRAアダプター付きモデル
    """
    # ベースモデルを凍結
    freeze_base_model(model)

    # LoRA設定
    config = StyleLoRAConfig(
        rank=lora_cfg.rank,
        alpha=lora_cfg.alpha,
        target_modules=list(lora_cfg.target_modules),
        dropout=lora_cfg.dropout,
    )

    # トランスフォーマーにLoRAを追加
    lora_config = config.to_lora_config()
    model.transformer = get_peft_model(model.transformer, lora_config)

    # LoRAパラメータのみ訓練可能に
    unfreeze_lora_params(model.transformer)

    # パラメータ数を表示
    trainable, total = count_trainable_params(model)
    print(f"訓練可能パラメータ: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    return model


class StyleLoRATrainer(Trainer):
    """Style LoRA訓練用のTrainerサブクラス"""

    def __init__(self, *args, style_attribute: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.style_attribute = style_attribute

    def save_checkpoint(self, step, last=False):
        """LoRAのみを保存"""
        if last:
            save_path = Path(self.checkpoint_path) / f"lora_{self.style_attribute}_last.safetensors"
        else:
            save_path = Path(self.checkpoint_path) / f"lora_{self.style_attribute}_{step}.safetensors"

        save_path.parent.mkdir(parents=True, exist_ok=True)

        # LoRAパラメータのみ抽出
        lora_state_dict = {}
        for name, param in self.model.transformer.named_parameters():
            if "lora_" in name:
                lora_state_dict[name] = param.cpu()

        # safetensorsで保存
        from safetensors.torch import save_file
        save_file(lora_state_dict, save_path)
        print(f"LoRAチェックポイントを保存: {save_path}")


@hydra.main(
    version_base="1.3",
    config_path=str(files("f5_tts").joinpath("configs")),
    config_name="ReStyleTTS_Base",
)
def main(cfg: DictConfig):
    """Style LoRA訓練のメイン関数"""

    # スタイル属性の検証
    style_attribute = cfg.get("style_attribute", None)
    if style_attribute is None:
        raise ValueError("style_attribute を指定してください。例: style_attribute=pitch_high")

    if style_attribute not in STYLE_ATTRIBUTES:
        raise ValueError(
            f"不明なスタイル属性: {style_attribute}. "
            f"利用可能な属性: {list(STYLE_ATTRIBUTES.keys())}"
        )

    print(f"=== Style LoRA 訓練 ===")
    print(f"スタイル属性: {style_attribute} ({STYLE_ATTRIBUTES[style_attribute]})")

    # モデル設定
    mel_spec_type = cfg.model.mel_spec.mel_spec_type
    exp_name = f"StyleLoRA_{style_attribute}_{mel_spec_type}"

    # ベースモデル読み込み
    print("ベースモデルを読み込み中...")
    model = load_base_model(cfg)

    # 事前学習済み重みを読み込み
    if cfg.get("pretrained_checkpoint", None):
        load_pretrained_weights(model, cfg.pretrained_checkpoint)

    # LoRAアダプターを追加
    print("LoRAアダプターを追加中...")
    model = setup_lora_model(model, cfg.lora)

    # Trainer初期化
    trainer = StyleLoRATrainer(
        model,
        style_attribute=style_attribute,
        epochs=cfg.optim.epochs,
        learning_rate=cfg.optim.get("lora_learning_rate", cfg.optim.learning_rate),
        num_warmup_updates=cfg.optim.num_warmup_updates,
        save_per_updates=cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=str(files("f5_tts").joinpath(f"../../{cfg.ckpts.save_dir}/lora_{style_attribute}")),
        batch_size_per_gpu=cfg.datasets.batch_size_per_gpu,
        batch_size_type=cfg.datasets.batch_size_type,
        max_samples=cfg.datasets.max_samples,
        grad_accumulation_steps=cfg.optim.grad_accumulation_steps,
        max_grad_norm=cfg.optim.max_grad_norm,
        logger=cfg.ckpts.logger,
        wandb_project="ReStyle-TTS-LoRA",
        wandb_run_name=exp_name,
        last_per_updates=cfg.ckpts.last_per_updates,
        log_samples=cfg.ckpts.log_samples,
        bnb_optimizer=cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=cfg.model.vocoder.is_local,
        local_vocoder_path=cfg.model.vocoder.local_path,
        model_cfg_dict=OmegaConf.to_container(cfg, resolve=True),
    )

    # データセット読み込み
    print("データセットを読み込み中...")
    train_dataset = load_dataset(
        cfg.datasets.name,
        cfg.model.tokenizer,
        mel_spec_kwargs=cfg.model.mel_spec,
    )

    # 訓練開始
    print("訓練を開始...")
    trainer.train(
        train_dataset,
        num_workers=cfg.datasets.num_workers,
        resumable_with_seed=666,
    )


if __name__ == "__main__":
    main()
