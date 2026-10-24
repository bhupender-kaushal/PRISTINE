#!/usr/bin/env python
"""Generate images with the paper-aligned CSGen ControlNet and HPCS conditioner."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from accelerate.utils import set_seed
from PIL import Image, ImageOps

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    SD3ControlNetModel,
    StableDiffusion3ControlNetPipeline,
)

from csgen.models.hpcs import HPCSConditioner
from csgen.utils.prompts import ensure_domain_prefix
from csgen.utils.text_encoding import encode_prompt_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate with SD3.5 + the trained CSGen ControlNet (no metric computation)."
    )
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument(
        "--trained_model_dir",
        required=True,
        help="Training output containing controlnet/ and hpcs/ subdirectories.",
    )
    parser.add_argument("--control_image", required=True)
    parser.add_argument("--domain", required=True, help="Domain name used in the explicit [DOMAIN=...] prefix.")
    parser.add_argument("--rule_prompt", required=True, help="Geometric/rule prompt encoded by both CLIP encoders.")
    parser.add_argument("--vlm_prompt", required=True, help="Detailed contextual prompt encoded by T5.")
    parser.add_argument("--negative_rule_prompt", default="")
    parser.add_argument("--negative_vlm_prompt", default="")
    parser.add_argument("--output", required=True, help="Output image path; use a .png extension.")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=28,
        help="FlowMatchEuler steps. The paper does not disclose this value.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.0,
        help="Classifier-free guidance scale. The paper does not disclose this value.",
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=1.0,
        help="ControlNet strength. The paper does not disclose this value.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--no_cpu_offload",
        action="store_true",
        help="Keep the full pipeline on the selected device instead of using model CPU offload.",
    )
    args = parser.parse_args()

    if args.height <= 0 or args.width <= 0 or args.height % 8 or args.width % 8:
        parser.error("--height and --width must be positive multiples of 8")
    if not 1 <= args.max_sequence_length <= 512:
        parser.error("--max_sequence_length must be in [1, 512]")
    if args.num_inference_steps <= 0:
        parser.error("--num_inference_steps must be positive")
    if args.controlnet_conditioning_scale < 0:
        parser.error("--controlnet_conditioning_scale must be non-negative")
    return args


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def main() -> None:
    args = parse_args()
    # The pipeline also samples the control-image VAE posterior using global RNG.
    set_seed(args.seed)
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Pass --device cpu --dtype fp32 if needed.")
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("fp16 inference on CPU is unsupported; use --dtype fp32 or bf16.")

    trained_model_dir = Path(args.trained_model_dir).expanduser().resolve()
    controlnet_dir = trained_model_dir / "controlnet"
    hpcs_dir = trained_model_dir / "hpcs"
    if not controlnet_dir.is_dir() or not hpcs_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {controlnet_dir} and {hpcs_dir}; point --trained_model_dir at a CSGen training output."
        )

    controlnet = SD3ControlNetModel.from_pretrained(controlnet_dir, torch_dtype=dtype)
    pipeline = StableDiffusion3ControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet,
        torch_dtype=dtype,
    )
    if not isinstance(pipeline.scheduler, FlowMatchEulerDiscreteScheduler):
        raise TypeError(
            "The paper requires FlowMatchEulerDiscreteScheduler, but the base model loaded "
            f"{type(pipeline.scheduler).__name__}."
        )

    if args.no_cpu_offload:
        pipeline.to(device)
    else:
        if device.type != "cuda":
            raise ValueError("Model CPU offload currently requires a CUDA device; pass --no_cpu_offload.")
        pipeline.enable_model_cpu_offload(device=device)
    pipeline.set_progress_bar_config(disable=False)

    hpcs_conditioner = HPCSConditioner.from_pretrained(hpcs_dir, device=device, dtype=dtype)
    hpcs_conditioner.eval()

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
    with torch.inference_mode():
        rule_prompt = ensure_domain_prefix(args.rule_prompt, args.domain)
        vlm_prompt = ensure_domain_prefix(args.vlm_prompt, args.domain)
        clip_hidden, pooled_hidden, t5_hidden = encode_prompt_components(
            text_encoders,
            tokenizers,
            clip_prompt=[rule_prompt],
            t5_prompt=[vlm_prompt],
            max_sequence_length=args.max_sequence_length,
            device=device,
        )
        negative_clip, negative_pooled, negative_t5 = encode_prompt_components(
            text_encoders,
            tokenizers,
            clip_prompt=[args.negative_rule_prompt],
            t5_prompt=[args.negative_vlm_prompt],
            max_sequence_length=args.max_sequence_length,
            device=device,
        )
        # Training reaches lambda=1 at 80%; inference uses the fully injected context branch.
        prompt_embeds = hpcs_conditioner(clip_hidden, t5_hidden, context_weight=1.0)
        negative_prompt_embeds = hpcs_conditioner(negative_clip, negative_t5, context_weight=1.0)

        control_image = Image.open(args.control_image).convert("RGB")
        control_image = ImageOps.fit(
            control_image, (args.width, args.height), method=Image.Resampling.BILINEAR
        )
        generator_device = device if device.type == "cuda" else torch.device("cpu")
        generator = torch.Generator(device=generator_device).manual_seed(args.seed)
        image = pipeline(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_hidden,
            negative_pooled_prompt_embeds=negative_pooled,
            control_image=control_image,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            generator=generator,
        ).images[0]

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved generated image to {output_path}")


if __name__ == "__main__":
    main()
