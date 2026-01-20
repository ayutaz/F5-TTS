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
        """LoRA重みと訓練状態を保存"""
        save_path = Path(self.checkpoint_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # DDPラッパーを外してモデルにアクセス
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # LoRAパラメータのみ抽出
        lora_state_dict = {}
        for name, param in unwrapped_model.transformer.named_parameters():
            if "lora_" in name:
                lora_state_dict[name] = param.cpu().clone()

        # LoRA重みをsafetensorsで保存（推論用）
        from safetensors.torch import save_file
        if last:
            lora_file = save_path / f"lora_{self.style_attribute}_last.safetensors"
        else:
            lora_file = save_path / f"lora_{self.style_attribute}_{step}.safetensors"
        save_file(lora_state_dict, lora_file)
        print(f"LoRAチェックポイントを保存: {lora_file}")

        # 訓練状態を保存（resume用）
        training_state = {
            "update": step,
            "lora_state_dict": lora_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scheduler:
            training_state["scheduler_state_dict"] = self.scheduler.state_dict()

        if last:
            state_file = save_path / f"training_state_last.pt"
        else:
            state_file = save_path / f"training_state_{step}.pt"
        torch.save(training_state, state_file)
        print(f"訓練状態を保存: {state_file}")

    def load_checkpoint(self):
        """訓練状態を読み込んでresumeする"""
        save_path = Path(self.checkpoint_path)

        if not save_path.exists():
            return 0

        # training_state_*.pt を探す
        state_files = list(save_path.glob("training_state_*.pt"))
        if not state_files:
            return 0

        # 最新のファイルを選択（lastを優先、なければ最大のstep）
        last_file = save_path / "training_state_last.pt"
        if last_file.exists():
            latest_file = last_file
        else:
            # stepの数値でソート
            def get_step(f):
                try:
                    return int(f.stem.split("_")[-1])
                except ValueError:
                    return 0
            latest_file = max(state_files, key=get_step)

        print(f"訓練状態を読み込み: {latest_file}")

        checkpoint = torch.load(latest_file, map_location="cpu", weights_only=False)

        # LoRA重みを読み込み
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        lora_state = checkpoint["lora_state_dict"]
        current_state = dict(unwrapped_model.transformer.named_parameters())
        for name, param in lora_state.items():
            if name in current_state:
                current_state[name].data.copy_(param)
        print(f"LoRA重みを読み込みました")

        # Optimizer状態を読み込み（存在する場合のみ）
        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"Optimizer状態を読み込みました")
        else:
            print(f"Optimizer状態なし - 新しいoptimizerで再開")

        # Scheduler状態を読み込み
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"Scheduler状態を読み込みました")

        update = checkpoint.get("update", 0)
        print(f"Update {update} から再開します")

        return update


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
