# ruff: noqa: E402
"""
ReStyle-TTS Gradio Inference Script

Style LoRA を読み込み、スタイル制御付きの音声合成を行うGradio UI。

Usage:
    uv run python -m f5_tts.infer.infer_restyle_gradio --share
"""

import gc
import json
import os
import tempfile
from importlib.resources import files
from pathlib import Path

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio
from cached_path import cached_path

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
    tempfile_kwargs,
)
from f5_tts.model import DiT
from f5_tts.restyle.style_lora import StyleLoRAManager, StyleLoRAConfig


# Default paths for LoRA checkpoints
LORA_CHECKPOINT_PATHS = {
    "pitch_high": "ckpts/ReStyleTTS_Base_vocos_char_pitch_high/lora_pitch_high/lora_pitch_high_last.safetensors",
    "pitch_low": "ckpts/ReStyleTTS_Base_vocos_char_pitch_low/lora_pitch_low/lora_pitch_low_last.safetensors",
    "energy_high": "ckpts/ReStyleTTS_Base_vocos_char_energy_high/lora_energy_high/lora_energy_high_last.safetensors",
    "energy_low": "ckpts/ReStyleTTS_Base_vocos_char_energy_low/lora_energy_low/lora_energy_low_last.safetensors",
}

DEFAULT_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Global variables
vocoder = None
ema_model = None
style_lora_manager = None


def load_models():
    """Load base model, vocoder, and Style LoRA adapters"""
    global vocoder, ema_model, style_lora_manager

    print("Loading vocoder...")
    vocoder = load_vocoder()

    print("Loading F5-TTS base model...")
    ckpt_path = str(cached_path(DEFAULT_TTS_MODEL_CFG[0]))
    F5TTS_model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    ema_model = load_model(DiT, F5TTS_model_cfg, ckpt_path)

    print("Creating Style LoRA Manager...")
    # LoRA設定（訓練時と同じ設定を使用）
    lora_config = StyleLoRAConfig(
        rank=32,
        alpha=64,
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.ff.0.0", "ff.ff.2"],
        dropout=0.0,
    )
    style_lora_manager = StyleLoRAManager(ema_model.transformer, config=lora_config)

    # Load available LoRA checkpoints
    base_path = Path(str(files("f5_tts").joinpath("../..")))
    loaded_styles = []

    for style_name, rel_path in LORA_CHECKPOINT_PATHS.items():
        ckpt_path = base_path / rel_path
        if ckpt_path.exists():
            try:
                style_lora_manager.load_lora(style_name, ckpt_path)
                loaded_styles.append(style_name)
                print(f"  Loaded LoRA: {style_name}")
            except Exception as e:
                print(f"  Failed to load {style_name}: {e}")
        else:
            print(f"  LoRA not found: {style_name} ({ckpt_path})")

    print(f"\nLoaded {len(loaded_styles)} Style LoRA adapters: {loaded_styles}")
    return loaded_styles


def infer(
    ref_audio_orig,
    ref_text,
    gen_text,
    remove_silence,
    seed,
    cross_fade_duration=0.15,
    nfe_step=32,
    speed=1,
    show_info=gr.Info,
    # ReStyle-TTS parameters
    use_dcfg=False,
    lambda_t=2.0,
    lambda_a=0.5,
    style_weights=None,
    use_olora=True,
):
    """Inference function with Style LoRA support"""
    global ema_model, vocoder, style_lora_manager

    if not ref_audio_orig:
        gr.Warning("Please provide reference audio.")
        return gr.update(), gr.update(), ref_text

    # Set inference seed
    if seed < 0 or seed > 2**31 - 1:
        gr.Warning("Seed must in range 0 ~ 2147483647. Using random seed instead.")
        seed = np.random.randint(0, 2**31 - 1)
    torch.manual_seed(seed)
    used_seed = seed

    if not gen_text.strip():
        gr.Warning("Please enter text to generate or upload a text file.")
        return gr.update(), gr.update(), ref_text

    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
        show_info=show_info,
        progress=gr.Progress(),
        # ReStyle-TTS parameters
        use_dcfg=use_dcfg,
        lambda_t=lambda_t,
        lambda_a=lambda_a,
        style_weights=style_weights,
        use_olora=use_olora,
        style_lora_manager=style_lora_manager,
    )

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(suffix=".wav", **tempfile_kwargs) as f:
            temp_path = f.name
        try:
            sf.write(temp_path, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        finally:
            os.unlink(temp_path)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Save the spectrogram
    with tempfile.NamedTemporaryFile(suffix=".png", **tempfile_kwargs) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
    save_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave), spectrogram_path, ref_text, used_seed


def create_gradio_app(loaded_styles):
    """Create Gradio interface"""

    with gr.Blocks(title="ReStyle-TTS Demo") as app:
        gr.Markdown(
            """
# ReStyle-TTS Demo

**ReStyle-TTS** (arXiv:2601.03632) による連続スタイル制御付き音声合成デモです。

## 読み込み済みスタイル LoRA
"""
            + "\n".join([f"- **{s}**" for s in loaded_styles])
            + """

## 使い方
1. 参照音声をアップロード
2. 生成するテキストを入力
3. スタイルスライダーを調整
4. 「Synthesize」をクリック
"""
        )

        # Default reference audio path
        default_ref_audio = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
        ref_audio_input = gr.Audio(
            label="Reference Audio (参照音声)",
            type="filepath",
            value=default_ref_audio,
        )

        with gr.Row():
            gen_text_input = gr.Textbox(
                label="Text to Generate (生成テキスト)",
                lines=5,
                placeholder="生成したいテキストを入力してください...",
                scale=4,
            )

        generate_btn = gr.Button("Synthesize (音声合成)", variant="primary", size="lg")

        with gr.Accordion("Advanced Settings (詳細設定)", open=False):
            with gr.Row():
                ref_text_input = gr.Textbox(
                    label="Reference Text (参照テキスト)",
                    info="空欄の場合は自動文字起こしされます",
                    lines=2,
                    scale=4,
                )
            with gr.Row():
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=True, scale=2)
                seed_input = gr.Number(label="Seed", value=0, precision=0, scale=2)
                remove_silence = gr.Checkbox(label="Remove Silences", value=False, scale=2)
            with gr.Row():
                speed_slider = gr.Slider(
                    label="Speed (速度)", minimum=0.3, maximum=2.0, value=1.0, step=0.1
                )
                nfe_slider = gr.Slider(
                    label="NFE Steps", minimum=4, maximum=64, value=32, step=2
                )
                cross_fade_slider = gr.Slider(
                    label="Cross-Fade (s)", minimum=0.0, maximum=1.0, value=0.15, step=0.01
                )

        # ReStyle-TTS Settings
        with gr.Accordion("ReStyle Settings (スタイル設定)", open=True):
            gr.Markdown("### DCFG (Decoupled Classifier-Free Guidance)")
            with gr.Row():
                use_dcfg = gr.Checkbox(
                    label="Enable DCFG (DCFGを有効化)",
                    value=False,
                    info="テキストと参照音声のガイダンスを分離",
                )
            with gr.Row():
                lambda_t_slider = gr.Slider(
                    label="λ_t (Text Guidance)",
                    minimum=0.0, maximum=4.0, value=2.0, step=0.1,
                    info="テキストの影響度",
                )
                lambda_a_slider = gr.Slider(
                    label="λ_a (Audio Guidance)",
                    minimum=0.0, maximum=2.0, value=0.5, step=0.1,
                    info="参照音声の影響度",
                )

            gr.Markdown("---")
            gr.Markdown("### Style Control (スタイル制御)")

            with gr.Row():
                pitch_slider = gr.Slider(
                    label="Pitch (ピッチ)",
                    minimum=-2.0, maximum=2.0, value=0.0, step=0.1,
                    info="負:低い ← 0:変更なし → 正:高い",
                )
                energy_slider = gr.Slider(
                    label="Energy (エネルギー)",
                    minimum=-2.0, maximum=2.0, value=0.0, step=0.1,
                    info="負:弱い ← 0:変更なし → 正:強い",
                )
            with gr.Row():
                use_olora = gr.Checkbox(
                    label="Use OLoRA Fusion",
                    value=True,
                    info="複数スタイル適用時の干渉を軽減",
                )

        audio_output = gr.Audio(label="Synthesized Audio (合成音声)")
        spectrogram_output = gr.Image(label="Spectrogram")

        def basic_tts(
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            remove_silence,
            randomize_seed,
            seed_input,
            cross_fade_slider,
            nfe_slider,
            speed_slider,
            use_dcfg,
            lambda_t_slider,
            lambda_a_slider,
            pitch_slider,
            energy_slider,
            use_olora,
        ):
            if randomize_seed:
                seed_input = np.random.randint(0, 2**31 - 1)

            # Build style_weights dict from sliders
            style_weights = {}
            if pitch_slider != 0.0:
                if pitch_slider > 0:
                    style_weights["pitch_high"] = pitch_slider
                else:
                    style_weights["pitch_low"] = -pitch_slider
            if energy_slider != 0.0:
                if energy_slider > 0:
                    style_weights["energy_high"] = energy_slider
                else:
                    style_weights["energy_low"] = -energy_slider

            # Only pass style_weights if not empty
            final_style_weights = style_weights if style_weights else None

            audio_out, spectrogram_path, ref_text_out, used_seed = infer(
                ref_audio_input,
                ref_text_input,
                gen_text_input,
                remove_silence,
                seed=seed_input,
                cross_fade_duration=cross_fade_slider,
                nfe_step=nfe_slider,
                speed=speed_slider,
                use_dcfg=use_dcfg,
                lambda_t=lambda_t_slider,
                lambda_a=lambda_a_slider,
                style_weights=final_style_weights,
                use_olora=use_olora,
            )
            return audio_out, spectrogram_path, ref_text_out, used_seed

        generate_btn.click(
            basic_tts,
            inputs=[
                ref_audio_input,
                ref_text_input,
                gen_text_input,
                remove_silence,
                randomize_seed,
                seed_input,
                cross_fade_slider,
                nfe_slider,
                speed_slider,
                use_dcfg,
                lambda_t_slider,
                lambda_a_slider,
                pitch_slider,
                energy_slider,
                use_olora,
            ],
            outputs=[audio_output, spectrogram_output, ref_text_input, seed_input],
        )

        gr.Markdown(
            """
---
## Credits
- **F5-TTS**: [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)
- **ReStyle-TTS**: arXiv:2601.03632
"""
        )

    return app


@click.command()
@click.option("--port", "-p", default=7860, type=int, help="Port to run the app on")
@click.option("--host", "-H", default="0.0.0.0", help="Host to run the app on")
@click.option("--share", "-s", is_flag=True, default=False, help="Create public link")
def main(port, host, share):
    """ReStyle-TTS Gradio Demo"""
    print("=" * 60)
    print("ReStyle-TTS Demo")
    print("=" * 60)

    # Load models
    loaded_styles = load_models()

    if not loaded_styles:
        print("\nWarning: No Style LoRA adapters loaded!")
        print("Style control will have no effect.")

    # Create and launch app
    app = create_gradio_app(loaded_styles)

    print(f"\nStarting Gradio app on {host}:{port}")
    if share:
        print("Creating public link...")

    app.queue().launch(
        server_name=host,
        server_port=port,
        share=share,
    )


if __name__ == "__main__":
    main()
