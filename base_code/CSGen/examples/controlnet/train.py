#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import functools
import gc
import json
import logging
import math
import os
import random
import shutil

# Add repo root to path to import from tests
from datetime import timedelta
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, ProjectConfiguration, set_seed
from datasets import Dataset, DatasetDict, Image as DatasetImage, load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3ControlNetModel,
    SD3Transformer2DModel,
    StableDiffusion3ControlNetPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import free_memory
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import backend_empty_cache, is_compiled_module
from local_processed_sampling import prepare_local_processed_training_view

from csgen.utils.flow import flow_interpolate, sample_uniform_t, scheduler_timestep, velocity_target
from csgen.models.hpcs import HPCSConditioner, compute_hpcs_lambda
from csgen.utils.prompts import ensure_domain_prefix, prompt_domain
from csgen.utils.slrm import weighted_velocity_mse
from csgen.utils.text_encoding import encode_prompt_components


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.38.0.dev0")

logger = get_logger(__name__)


def _coerce_prompt_value(value, column_name: str, is_train: bool = True) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, np.ndarray)):
        if len(value) == 0:
            return ""
        return random.choice(value) if is_train else value[0]
    if value is None:
        return ""
    raise ValueError(f"Caption column `{column_name}` should contain either strings or lists of strings.")


def _coerce_float_value(value, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)


def _coerce_dict_value(value) -> dict:
    return value if isinstance(value, dict) else {}


def _prompt_from_meta(prompt_meta: dict, keys: tuple[str, ...], fallback: str, column_name: str, is_train: bool) -> str:
    for key in keys:
        if key not in prompt_meta:
            continue
        value = prompt_meta.get(key)
        if value is None:
            continue
        return _coerce_prompt_value(value, f"{column_name}.{key}", is_train=is_train)
    return fallback


def _resolve_caption_base_prompt(value, column_name: str, is_train: bool) -> tuple[str, dict]:
    if isinstance(value, dict):
        prompt_meta = value
        base_prompt = _prompt_from_meta(prompt_meta, ("template", "attribute"), "", column_name, is_train)
        return base_prompt, prompt_meta
    return _coerce_prompt_value(value, column_name, is_train=is_train), {}


def _has_prompt_value(value) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, np.ndarray)):
        return len(value) > 0 and any(str(item).strip() for item in value)
    return value is not None


def _parse_column_names(value: str | None) -> list[str]:
    if value is None:
        return []
    return [column.strip() for column in value.split(",") if column.strip()]


def _local_processed_domain_dirs(root: Path) -> list[Path]:
    if (root / "domain_rules.json").exists():
        return [root]
    return sorted(
        path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".") and (path / "domain_rules.json").exists()
    )


def _is_local_processed_dataset(root: Path) -> bool:
    return root.is_dir() and len(_local_processed_domain_dirs(root)) > 0


def _resolve_local_sample_dir(root: Path, sample: dict) -> Path:
    sample_id = str(sample.get("id", ""))
    if (root / "domain_rules.json").exists():
        return root / sample_id
    domain = str(sample.get("domain", ""))
    return root / domain / sample_id


def _load_local_processed_dataset(root: Path) -> DatasetDict:
    domain_dirs = _local_processed_domain_dirs(root)
    records = []

    for domain_dir in domain_dirs:
        for sample_json_path in sorted(domain_dir.glob("sample_*/*.json")):
            with sample_json_path.open("r", encoding="utf-8") as f:
                sample = json.load(f)

            files = _coerce_dict_value(sample.get("files", {}))
            if bool(sample.get("invalid_sample", False)):
                continue
            if "image" not in files:
                continue

            record = {
                "id": sample.get("id", sample_json_path.stem),
                "domain": sample.get("domain", domain_dir.name),
                "class_name": sample.get("class_name", sample.get("domain", domain_dir.name)),
                "invalid_sample": False,
                "files": files,
                "geometry_raw": _coerce_dict_value(sample.get("geometry_raw", {})),
                "appearance_raw": _coerce_dict_value(sample.get("appearance_raw", {})),
                "scale_meta": _coerce_dict_value(sample.get("scale_meta", {})),
                "attribute_bins": _coerce_dict_value(sample.get("attribute_bins", {})),
                "prompts": _coerce_dict_value(sample.get("prompts", {})),
                "clip_prompt": sample.get("clip_prompt", ""),
                "t5_prompt": sample.get("t5_prompt", ""),
                "derivation": _coerce_dict_value(sample.get("derivation", {})),
                "prompt_source": _coerce_dict_value(sample.get("prompt_source", {})),
            }
            records.append(record)

    if len(records) == 0:
        raise ValueError(f"No valid samples found under processed dataset root: {root}")

    logging.getLogger(__name__).info(
        "Loaded local processed dataset metadata from %s with %d samples",
        root,
        len(records),
    )
    return DatasetDict({"train": Dataset.from_list(records)})


def _attach_local_processed_file_columns(train_split: Dataset, root: Path) -> tuple[Dataset, list[str]]:
    def attach_paths(example):
        sample_dir = _resolve_local_sample_dir(root, example)
        files = _coerce_dict_value(example.get("files", {}))
        derived = {}
        for key, filename in files.items():
            if not isinstance(filename, str) or not filename:
                continue
            path = sample_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing file referenced by sample JSON: {path}")
            derived[key] = str(path)
        return derived

    train_split = train_split.map(attach_paths)
    if len(train_split) == 0:
        return train_split, []

    file_columns = sorted(_coerce_dict_value(train_split[0].get("files", {})).keys())
    for column in file_columns:
        if column in train_split.column_names:
            train_split = train_split.cast_column(column, DatasetImage())

    return train_split, file_columns


def log_validation(
    controlnet,
    hpcs_conditioner,
    transformer,
    vae,
    args,
    accelerator,
    weight_dtype,
    step,
    is_final_validation=False,
):
    logger.info("Running validation... ")

    controlnet = accelerator.unwrap_model(controlnet)
    controlnet_was_training = controlnet.training
    controlnet.eval()
    trained_hpcs = accelerator.unwrap_model(hpcs_conditioner)
    validation_hpcs = HPCSConditioner(
        clip_dim=trained_hpcs.clip_dim,
        t5_dim=trained_hpcs.t5_dim,
        joint_dim=trained_hpcs.joint_dim,
    )
    validation_hpcs.load_state_dict(
        {name: value.detach().cpu() for name, value in trained_hpcs.state_dict().items()}
    )
    hpcs_conditioner = validation_hpcs.eval()

    # First load only the frozen text encoders. This keeps validation memory bounded
    # while still applying the same paper Eq. (4) conditioner used for training.
    pipeline = StableDiffusion3ControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=None,
        transformer=None,
        vae=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(torch.device(accelerator.device))
    pipeline.set_progress_bar_config(disable=True)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    validation_images = args.validation_image
    validation_domains = args.validation_domain
    validation_prompts = [
        ensure_domain_prefix(prompt, domain)
        for prompt, domain in zip(args.validation_prompt, validation_domains)
    ]
    validation_vlm_prompts = [
        ensure_domain_prefix(prompt, domain)
        for prompt, domain in zip(args.validation_vlm_prompt, validation_domains)
    ]

    with torch.no_grad():
        clip_prompt_embeds, pooled_prompt_embeds, t5_prompt_embeds = encode_prompt_components(
            text_encoders=[pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3],
            tokenizers=[pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3],
            clip_prompt=validation_prompts,
            t5_prompt=validation_vlm_prompts,
            max_sequence_length=args.max_sequence_length,
            device=accelerator.device,
        )
        empty_prompts = [""] * len(validation_prompts)
        negative_clip_embeds, negative_pooled_prompt_embeds, negative_t5_embeds = encode_prompt_components(
            text_encoders=[pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3],
            tokenizers=[pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3],
            clip_prompt=empty_prompts,
            t5_prompt=empty_prompts,
            max_sequence_length=args.max_sequence_length,
            device=accelerator.device,
        )
        hpcs_conditioner.to(device=accelerator.device, dtype=weight_dtype)
        prompt_embeds = hpcs_conditioner(clip_prompt_embeds, t5_prompt_embeds, context_weight=1.0)
        negative_prompt_embeds = hpcs_conditioner(
            negative_clip_embeds, negative_t5_embeds, context_weight=1.0
        )
        hpcs_conditioner.to("cpu")

    del pipeline
    gc.collect()
    backend_empty_cache(accelerator.device.type)

    pipeline = StableDiffusion3ControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet,
        transformer=transformer,
        vae=vae,
        text_encoder=None,
        text_encoder_2=None,
        text_encoder_3=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    # Reuse training models without installing offload hooks or duplicating the backbone.
    pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    image_logs = []

    for i, validation_image in enumerate(validation_images):
        validation_image = Image.open(validation_image).convert("RGB")
        validation_prompt = validation_prompts[i]
        validation_vlm_prompt = validation_vlm_prompts[i]

        images = []

        for _ in range(args.num_validation_images):
            with accelerator.autocast():
                image = pipeline(
                    prompt_embeds=prompt_embeds[i].unsqueeze(0),
                    negative_prompt_embeds=negative_prompt_embeds[i].unsqueeze(0),
                    pooled_prompt_embeds=pooled_prompt_embeds[i].unsqueeze(0),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds[i].unsqueeze(0),
                    control_image=validation_image,
                    height=args.resolution,
                    width=args.resolution,
                    max_sequence_length=args.max_sequence_length,
                    num_inference_steps=args.validation_num_inference_steps,
                    generator=generator,
                ).images[0]

            images.append(image)

        image_logs.append(
            {
                "validation_image": validation_image,
                "images": images,
                "validation_prompt": validation_prompt,
                "validation_vlm_prompt": validation_vlm_prompt,
            }
        )

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_vlm_prompt = log["validation_vlm_prompt"]
                validation_image = log["validation_image"]

                tracker.writer.add_image(
                    "Controlnet conditioning", np.asarray([validation_image]), step, dataformats="NHWC"
                )

                formatted_images = []
                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(
                    f"{validation_prompt} | {validation_vlm_prompt}",
                    formatted_images,
                    step,
                    dataformats="NHWC",
                )
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_vlm_prompt = log["validation_vlm_prompt"]
                validation_image = log["validation_image"]

                formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))

                for image in images:
                    image = wandb.Image(
                        image,
                        caption=f"rule={validation_prompt} | vlm={validation_vlm_prompt}",
                    )
                    formatted_images.append(image)

            tracker.log({tracker_key: formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline
    free_memory()

    controlnet.train(controlnet_was_training)

    return image_logs


# Copied from dreambooth sd3 example
def load_text_encoders(class_one, class_two, class_three):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


# Copied from dreambooth sd3 example
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            make_image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    model_description = f"""
# SD3 controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
The weights were trained using [ControlNet](https://github.com/lllyasviel/ControlNet) with the [SD3 diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/README_sd3.md).
{img_str}

Please adhere to the licensing terms as described `[here](https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/LICENSE)`.
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="openrail++",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "sd3",
        "sd3-diffusers",
        "controlnet",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Paper-aligned CSGen training: SD3.5, HPCS, and SLRM.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--num_extra_conditioning_channels",
        type=int,
        default=0,
        help="Number of extra conditioning channels for controlnet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Global seed used by the paper configuration.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=48)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--upcast_vae",
        action="store_true",
        help="Whether or not to upcast vae to fp32",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--ddp_timeout_seconds",
        type=int,
        default=7200,
        help=(
            "Timeout used by distributed collectives. Increase this when the main process spends a long time "
            "precomputing dataset features before the other ranks can leave `main_process_first()`."
        ),
    )
    parser.add_argument(
        "--slrm_gamma",
        type=float,
        default=1.0,
        help="Gamma in the paper's W(d)=gamma*log(1+Dmax/(d+epsilon)) equation.",
    )
    parser.add_argument(
        "--slrm_epsilon",
        type=float,
        default=1e-6,
        help="Positive epsilon in the paper's SLRM equation.",
    )
    parser.add_argument(
        "--slrm_dmax",
        type=float,
        default=512.0,
        help="Dmax in the paper's SLRM equation. The paper does not disclose its value.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="mask,width_map,sketch,edge,partial_sketch",
        help=(
            "The column of the dataset containing the ControlNet conditioning image. "
            "Use a comma-separated list to load multiple control images. "
            "For local processed datasets, leaving this as `conditioning_image` loads all control maps."
        ),
    )
    parser.add_argument(
        "--min_conditioning_images",
        type=int,
        default=1,
        help="Minimum number of conditioning images to sample for each training example.",
    )
    parser.add_argument(
        "--max_conditioning_images",
        type=int,
        default=1,
        help="Must be 1: the paper samples exactly one structural control map per example.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--hpcs_anchor_ratio",
        type=float,
        default=0.4,
        help="tau_anchor/tau_total in Eq. (4).",
    )
    parser.add_argument(
        "--hpcs_eta",
        type=float,
        default=0.4,
        help="eta in Eq. (4); anchor=0.4 and eta=0.4 reaches full T5 weight at 0.8.",
    )
    parser.add_argument(
        "--hpcs_power",
        type=float,
        default=1.0,
        help="Exponent k in Eq. (4). The paper does not disclose its value.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--enable_domain_rebalance",
        action="store_true",
        help="Use domain-temperature resampling instead of the raw domain frequency for local processed datasets.",
    )
    parser.add_argument(
        "--domain_balance_alpha",
        type=float,
        default=0.5,
        help="Temperature exponent for domain rebalancing, where q(d) is proportional to n_d^alpha.",
    )
    parser.add_argument(
        "--counterfactual_pair_prob",
        type=float,
        default=0.0,
        help=(
            "Probability of appending a matched cross-domain counterfactual sample with similar structure/layout "
            "metadata after each anchor sample in the resampled local processed dataset."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--dataset_preprocess_batch_size", type=int, default=1,
        help="T5 embedding precompute batch size; separate from training batch size."
    )
    parser.add_argument(
        "--validation_domain",
        type=str,
        default=None,
        nargs="+",
        help="Domain name for each validation prompt/control triplet.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "Rule/attribute prompts encoded by CLIP for validation. Provide one per validation image."
        ),
    )
    parser.add_argument(
        "--validation_vlm_prompt",
        type=str,
        default=None,
        nargs="+",
        help="VLM descriptions encoded by T5 for validation. Provide one per validation image.",
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "Control image paths used for validation. Provide one per CLIP/T5 prompt pair."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--validation_num_inference_steps",
        type=int,
        default=28,
        help="FlowMatchEuler denoising steps for validation; the paper does not disclose this value.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    locked_paper_values = {
        "seed": 2026,
        "train_batch_size": 1,
        "num_train_epochs": 48,
        "max_train_steps": None,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-6,
        "scale_lr": False,
        "lr_scheduler": "constant_with_warmup",
        "lr_warmup_steps": 500,
        "use_8bit_adam": False,
        "adam_weight_decay": 1e-2,
        "max_sequence_length": 512,
        "mixed_precision": "bf16",
        "checkpointing_steps": 500,
        "validation_steps": 500,
        "proportion_empty_prompts": 0.0,
        "max_train_samples": None,
        "hpcs_anchor_ratio": 0.4,
        "hpcs_eta": 0.4,
    }
    for name, expected in locked_paper_values.items():
        actual = getattr(args, name)
        values_match = (
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            if isinstance(expected, float)
            else actual == expected
        )
        if not values_match:
            raise ValueError(
                f"Paper-aligned mode locks --{name}={expected!r}; received {actual!r}."
            )

    expected_control_columns = {"mask", "width_map", "sketch", "edge", "partial_sketch"}
    parsed_control_columns = _parse_column_names(args.conditioning_image_column)
    actual_control_columns = set(parsed_control_columns)
    if actual_control_columns != expected_control_columns or len(parsed_control_columns) != 5:
        raise ValueError(
            "Paper-aligned mode requires exactly mask,width_map,sketch,edge,partial_sketch."
        )

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.min_conditioning_images != 1 or args.max_conditioning_images != 1:
        raise ValueError("Paper-aligned training requires exactly one randomly selected control map per sample.")

    if args.slrm_gamma <= 0 or args.slrm_epsilon <= 0 or args.slrm_dmax <= 0:
        raise ValueError("SLRM gamma, epsilon, and Dmax must all be positive.")
    if args.dataset_preprocess_batch_size <= 0:
        raise ValueError("--dataset_preprocess_batch_size must be positive.")
    if args.num_extra_conditioning_channels != 0:
        raise ValueError("This single-layout VAE conditioning path requires zero extra conditioning channels.")

    if args.domain_balance_alpha < 0:
        raise ValueError("`--domain_balance_alpha` must be non-negative.")

    if not 0.0 <= args.counterfactual_pair_prob <= 1.0:
        raise ValueError("`--counterfactual_pair_prob` must be in the range [0, 1].")

    if not 0.0 <= args.hpcs_anchor_ratio <= 1.0:
        raise ValueError("`--hpcs_anchor_ratio` must be in [0, 1].")
    if args.hpcs_eta <= 0.0 or args.hpcs_anchor_ratio + args.hpcs_eta > 1.0:
        raise ValueError("HPCS eta must be positive and anchor_ratio + eta must not exceed 1.")
    if args.hpcs_power <= 0.0:
        raise ValueError("`--hpcs_power` must be positive.")

    if args.enable_domain_rebalance or args.counterfactual_pair_prob != 0.0:
        raise ValueError("Domain rebalancing/counterfactual pairing is not part of the paper training contract.")

    validation_fields = (
        args.validation_image,
        args.validation_domain,
        args.validation_prompt,
        args.validation_vlm_prompt,
    )
    if any(value is not None for value in validation_fields) and not all(
        value is not None for value in validation_fields
    ):
        raise ValueError(
            "Validation requires all four of --validation_image, --validation_prompt, "
            "--validation_vlm_prompt, and --validation_domain."
        )
    if all(value is not None for value in validation_fields):
        validation_lengths = {len(value) for value in validation_fields}
        if len(validation_lengths) != 1:
            raise ValueError("Validation image, CLIP prompt, and T5 prompt counts must match exactly.")
    if args.validation_num_inference_steps <= 0:
        raise ValueError("--validation_num_inference_steps must be positive.")

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def make_train_dataset(args, tokenizer_one, tokenizer_two, tokenizer_three, accelerator):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    local_processed_dataset = False
    local_file_columns = []
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        if args.train_data_dir is not None:
            train_data_dir = Path(args.train_data_dir).expanduser().resolve()
            if _is_local_processed_dataset(train_data_dir):
                dataset = _load_local_processed_dataset(train_data_dir)
                dataset["train"], local_file_columns = _attach_local_processed_file_columns(
                    dataset["train"], train_data_dir
                )
                local_processed_dataset = True
                logger.info("Detected local processed dataset layout under %s", train_data_dir)
            else:
                dataset = load_dataset(
                    args.train_data_dir,
                    cache_dir=args.cache_dir,
                    trust_remote_code=True,
                )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            if local_processed_dataset and caption_column == "text" and "clip_prompt" in column_names:
                caption_column = "clip_prompt"
                logger.info(
                    "caption column `%s` not found for local processed dataset; remapping to `%s`",
                    args.caption_column,
                    caption_column,
                )
            elif local_processed_dataset and caption_column == "text" and "prompts" in column_names:
                caption_column = "prompts"
                logger.info(
                    "caption column `%s` not found for local processed dataset; remapping to `%s`",
                    args.caption_column,
                    caption_column,
                )
            else:
                raise ValueError(
                    f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

    if args.conditioning_image_column is None:
        conditioning_image_columns = [column_names[2]]
        logger.info(f"conditioning image column defaulting to {conditioning_image_columns[0]}")
    else:
        requested_conditioning_columns = _parse_column_names(args.conditioning_image_column)
        if len(requested_conditioning_columns) == 0:
            raise ValueError("`--conditioning_image_column` must specify at least one column name.")

        if local_processed_dataset and requested_conditioning_columns in (["conditioning_image"], ["all"]):
            conditioning_image_columns = [column for column in local_file_columns if column != image_column]
            if len(conditioning_image_columns) == 0:
                raise ValueError(
                    "No usable conditioning image columns were found in the local processed dataset. "
                    f"Available file columns are: {', '.join(local_file_columns)}"
                )
            logger.info(
                "conditioning image column `%s` remapped to local control columns: %s",
                args.conditioning_image_column,
                ", ".join(conditioning_image_columns),
            )
        else:
            missing_conditioning_columns = [
                column for column in requested_conditioning_columns if column not in column_names
            ]
            if len(missing_conditioning_columns) > 0:
                raise ValueError(
                    f"`--conditioning_image_column` values '{', '.join(requested_conditioning_columns)}' not found in dataset columns. "
                    f"Missing: {', '.join(missing_conditioning_columns)}. Dataset columns are: {', '.join(column_names)}"
                )

            if local_processed_dataset:
                non_file_columns = [column for column in requested_conditioning_columns if column not in local_file_columns]
                if len(non_file_columns) > 0:
                    raise ValueError(
                        "For the local processed dataset, `--conditioning_image_column` must reference control image files. "
                        f"Invalid columns: {', '.join(non_file_columns)}. "
                        f"Available control columns are: {', '.join(column for column in local_file_columns if column != image_column)}"
                    )
                if image_column in requested_conditioning_columns:
                    raise ValueError(
                        f"`{image_column}` is the target image column and cannot also be used as a conditioning image. "
                        f"Choose from: {', '.join(column for column in local_file_columns if column != image_column)}"
                    )

            conditioning_image_columns = requested_conditioning_columns

    max_conditioning_images = len(conditioning_image_columns)
    if args.max_conditioning_images is not None:
        max_conditioning_images = min(args.max_conditioning_images, len(conditioning_image_columns))

    if args.min_conditioning_images > len(conditioning_image_columns):
        raise ValueError(
            f"`--min_conditioning_images`={args.min_conditioning_images} exceeds the number of available conditioning columns ({len(conditioning_image_columns)}): "
            f"{', '.join(conditioning_image_columns)}"
        )

    if max_conditioning_images < 1:
        raise ValueError("No conditioning images are available after applying the current configuration.")

    min_conditioning_images = min(args.min_conditioning_images, max_conditioning_images)

    prompts_column = "prompts" if "prompts" in column_names else None
    clip_prompt_column = "clip_prompt" if "clip_prompt" in column_names else None
    t5_prompt_column = "t5_prompt" if "t5_prompt" in column_names else None
    scale_meta_column = "scale_meta" if "scale_meta" in column_names else None
    domain_column = "domain" if "domain" in column_names else None

    def process_captions(examples, is_train=True):
        prompt_inputs = examples[caption_column]
        prompt_meta_inputs = examples[prompts_column] if prompts_column is not None else [None] * len(prompt_inputs)
        clip_prompt_inputs = examples[clip_prompt_column] if clip_prompt_column is not None else [None] * len(prompt_inputs)
        t5_prompt_inputs = examples[t5_prompt_column] if t5_prompt_column is not None else [None] * len(prompt_inputs)
        scale_meta_inputs = examples[scale_meta_column] if scale_meta_column is not None else [None] * len(prompt_inputs)
        domain_inputs = examples[domain_column] if domain_column is not None else [None] * len(prompt_inputs)

        prompt_templates = []
        prompt_attributes = []
        equivalent_diameter_px_values = []

        for idx, prompt_value in enumerate(prompt_inputs):
            prompt_meta = _coerce_dict_value(prompt_meta_inputs[idx])
            base_prompt, base_prompt_meta = _resolve_caption_base_prompt(prompt_value, caption_column, is_train)
            if not prompt_meta:
                prompt_meta = base_prompt_meta

            if random.random() < args.proportion_empty_prompts:
                prompt_template = ""
                prompt_attribute = ""
            else:
                prompt_attribute = (
                    _coerce_prompt_value(clip_prompt_inputs[idx], clip_prompt_column or "clip_prompt", is_train)
                    if _has_prompt_value(clip_prompt_inputs[idx])
                    else ""
                )
                prompt_template = (
                    _coerce_prompt_value(t5_prompt_inputs[idx], t5_prompt_column or "t5_prompt", is_train)
                    if _has_prompt_value(t5_prompt_inputs[idx])
                    else ""
                )

                if not prompt_attribute:
                    prompt_attribute = _prompt_from_meta(
                        prompt_meta,
                        ("attribute", "template"),
                        base_prompt,
                        prompts_column or caption_column,
                        is_train,
                    )

                if not prompt_template:
                    prompt_template = _prompt_from_meta(
                        prompt_meta,
                        ("template", "attribute"),
                        prompt_attribute or base_prompt,
                        prompts_column or caption_column,
                        is_train,
                    )

                domain = str(domain_inputs[idx]).strip() if domain_inputs[idx] is not None else ""
                if domain:
                    prompt_attribute = ensure_domain_prefix(prompt_attribute, domain)
                    prompt_template = ensure_domain_prefix(prompt_template, domain)
                elif prompt_domain(prompt_attribute) is None or prompt_domain(prompt_template) is None:
                    raise ValueError(
                        "Paper-aligned prompts require an explicit [DOMAIN=...] prefix. "
                        "Provide a `domain` column or include the prefix in both CLIP and T5 prompts."
                    )
                if not prompt_attribute or not prompt_template:
                    raise ValueError("Both the CLIP rule prompt and T5 VLM prompt must be non-empty.")

            scale_meta = _coerce_dict_value(scale_meta_inputs[idx])
            equivalent_diameter_px = _coerce_float_value(scale_meta.get("equivalent_diameter_px", 0.0), 0.0)
            if equivalent_diameter_px <= 0.0:
                raise ValueError(
                    "Paper-aligned SLRM requires positive scale_meta.equivalent_diameter_px "
                    "computed from the original binary mask."
                )

            prompt_templates.append(prompt_template)
            prompt_attributes.append(prompt_attribute)
            equivalent_diameter_px_values.append(equivalent_diameter_px)

        return prompt_templates, prompt_attributes, equivalent_diameter_px_values

    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        images = [image_transforms(image) for image in images]

        conditioning_pixel_values = []
        conditioning_image_columns_per_example = []
        for example_idx in range(len(images)):
            sampled_conditioning_count = random.randint(min_conditioning_images, max_conditioning_images)
            sampled_conditioning_columns = random.sample(conditioning_image_columns, k=sampled_conditioning_count)
            per_example_conditioning_images = []
            for conditioning_column in sampled_conditioning_columns:
                conditioning_image = examples[conditioning_column][example_idx].convert("RGB")
                per_example_conditioning_images.append(conditioning_image_transforms(conditioning_image))
            conditioning_pixel_values.append(torch.stack(per_example_conditioning_images))
            conditioning_image_columns_per_example.append(sampled_conditioning_columns)

        prompt_template, prompt_attribute, equivalent_diameter_px = process_captions(examples)

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_pixel_values
        examples["conditioning_image_columns"] = conditioning_image_columns_per_example
        examples["prompt_template"] = prompt_template
        examples["prompt_attribute"] = prompt_attribute
        examples["equivalent_diameter_px"] = equivalent_diameter_px

        return examples

    train_dataset_stats = {}
    with accelerator.main_process_first():
        if local_processed_dataset and (args.enable_domain_rebalance or args.counterfactual_pair_prob > 0.0):
            dataset["train"], sampling_stats = prepare_local_processed_training_view(dataset["train"], args, logger=logger)
            train_dataset_stats["sampling"] = sampling_stats
        elif args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset, train_dataset_stats


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    max_conditioning_images = max(example["conditioning_pixel_values"].shape[0] for example in examples)
    conditioning_pixel_values = []

    for example in examples:
        example_conditioning = example["conditioning_pixel_values"]
        example_conditioning_count = example_conditioning.shape[0]
        pad_count = max_conditioning_images - example_conditioning_count

        if pad_count > 0:
            padding = torch.zeros(
                (pad_count, *example_conditioning.shape[1:]),
                dtype=example_conditioning.dtype,
            )
            example_conditioning = torch.cat([example_conditioning, padding], dim=0)

        conditioning_pixel_values.append(example_conditioning)

    conditioning_pixel_values = torch.stack(conditioning_pixel_values)
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    clip_prompt_embeds = torch.stack([torch.as_tensor(example["clip_prompt_embeds"]) for example in examples])
    t5_prompt_embeds = torch.stack([torch.as_tensor(example["t5_prompt_embeds"]) for example in examples])
    pooled_prompt_embeds = torch.stack([torch.as_tensor(example["pooled_prompt_embeds"]) for example in examples])
    equivalent_diameter_px = torch.tensor(
        [_coerce_float_value(example["equivalent_diameter_px"], 0.0) for example in examples]
    ).float()

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "clip_prompt_embeds": clip_prompt_embeds,
        "t5_prompt_embeds": t5_prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "equivalent_diameter_px": equivalent_diameter_px,
    }


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.ddp_timeout_seconds))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs, process_group_kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.revision,
    )

    # import correct text encoder class
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = SD3ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from transformer")
        controlnet = SD3ControlNetModel.from_transformer(
            transformer, num_layers=12,
            num_extra_conditioning_channels=args.num_extra_conditioning_channels
        )
    # CSGen uses the two CLIP global projections and the same VAE latent shift
    # for target and control images. Persist this so pipeline inference matches training.
    controlnet.register_to_config(force_zeros_for_pooled_projection=False)

    hpcs_conditioner = HPCSConditioner(
        clip_dim=int(text_encoder_one.config.hidden_size + text_encoder_two.config.hidden_size),
        t5_dim=int(
            text_encoder_three.config.d_model
            if hasattr(text_encoder_three.config, "d_model")
            else text_encoder_three.config.hidden_size
        ),
        joint_dim=int(transformer.config.joint_attention_dim),
    )

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)
    controlnet.train()
    hpcs_conditioner.train()

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for model in models:
                    model = unwrap_model(model)
                    if isinstance(model, SD3ControlNetModel):
                        model.save_pretrained(os.path.join(output_dir, "controlnet"))
                    elif isinstance(model, HPCSConditioner):
                        model.save_pretrained(os.path.join(output_dir, "hpcs"))
                    else:
                        raise TypeError(f"Unexpected trainable model in save hook: {type(model)!r}")
                while weights:
                    weights.pop()

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                model = models.pop()
                unwrapped = unwrap_model(model)
                if isinstance(unwrapped, SD3ControlNetModel):
                    load_model = SD3ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                    unwrapped.register_to_config(**load_model.config)
                    unwrapped.load_state_dict(load_model.state_dict())
                    del load_model
                elif isinstance(unwrapped, HPCSConditioner):
                    load_model = HPCSConditioner.from_pretrained(Path(input_dir) / "hpcs")
                    unwrapped.load_state_dict(load_model.state_dict())
                    del load_model
                else:
                    raise TypeError(f"Unexpected trainable model in load hook: {type(unwrapped)!r}")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
        transformer.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )
    if unwrap_model(hpcs_conditioner).phi.weight.dtype != torch.float32:
        raise ValueError("HPCS projections must start in float32 precision." + low_precision_error_string)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = list(controlnet.parameters()) + list(hpcs_conditioner.parameters())
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, transformer and text_encoder to device and cast to weight_dtype
    if args.upcast_vae:
        vae.to(accelerator.device, dtype=torch.float32)
    else:
        vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_three.to(accelerator.device, dtype=weight_dtype)

    train_dataset, train_dataset_stats = make_train_dataset(
        args, tokenizer_one, tokenizer_two, tokenizer_three, accelerator
    )
    if "sampling" in train_dataset_stats:
        logger.info("Using local processed sampling policy: %s", train_dataset_stats["sampling"])

    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]

    def compute_text_embeddings(batch, text_encoders, tokenizers):
        with torch.no_grad():
            clip_prompt = batch["prompt_attribute"] if "prompt_attribute" in batch else batch["prompts"]
            t5_prompt = batch["prompt_template"] if "prompt_template" in batch else clip_prompt
            clip_prompt_embeds, pooled_prompt_embeds, t5_prompt_embeds = encode_prompt_components(
                text_encoders=text_encoders,
                tokenizers=tokenizers,
                clip_prompt=clip_prompt,
                t5_prompt=t5_prompt,
                max_sequence_length=args.max_sequence_length,
            )
            clip_prompt_embeds = clip_prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            t5_prompt_embeds = t5_prompt_embeds.to(accelerator.device)
        return {
            "clip_prompt_embeds": clip_prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "t5_prompt_embeds": t5_prompt_embeds,
        }

    compute_embeddings_fn = functools.partial(
        compute_text_embeddings,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
    )
    with accelerator.main_process_first():
        from datasets.fingerprint import Hasher

        # fingerprint used by the cache for the other processes to load the result
        # details: https://github.com/huggingface/diffusers/pull/4038#discussion_r1266078401
        new_fingerprint = Hasher.hash({"args": vars(args), "embedding_contract": "csgen_additive_domain_prefix_v1"})
        map_kwargs = {
            "batched": True,
            "batch_size": args.dataset_preprocess_batch_size,
            "new_fingerprint": new_fingerprint,
        }
        if args.train_data_dir is not None and _is_local_processed_dataset(Path(args.train_data_dir).expanduser().resolve()):
            embedding_cache_dir = Path(args.train_data_dir).expanduser().resolve() / ".cache_text_embeddings"
            embedding_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file_name = embedding_cache_dir / f"text_embeds_{new_fingerprint}.arrow"
            map_kwargs["cache_file_name"] = str(cache_file_name)
            logger.info("Caching precomputed text embeddings to %s", cache_file_name)
        train_dataset = train_dataset.map(
            compute_embeddings_fn,
            **map_kwargs,
        )

    del text_encoder_one, text_encoder_two, text_encoder_three
    del tokenizer_one, tokenizer_two, tokenizer_three
    del compute_embeddings_fn, text_encoders, tokenizers
    free_memory()

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    controlnet, hpcs_conditioner, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, hpcs_conditioner, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_vlm_prompt")
        tracker_config.pop("validation_domain")
        tracker_config.pop("validation_image")
        tracker_config.pop("hub_token", None)

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet, hpcs_conditioner):
                # Convert images to latent space
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
                # Paper Eq. (2): t ~ U(0, 1), z_t = (1 - t) z_0 + t z_1.
                flow_t = sample_uniform_t(
                    bsz,
                    device=model_input.device,
                    dtype=torch.float32,
                )
                noisy_model_input = flow_interpolate(model_input, noise, flow_t)
                timesteps = scheduler_timestep(
                    flow_t,
                    noise_scheduler.config.num_train_timesteps,
                )

                # Paper Eq. (4): Phi(CLIP(rule)) + lambda * Psi(T5(VLM description)).
                clip_prompt_embeds = batch["clip_prompt_embeds"].to(dtype=weight_dtype)
                t5_prompt_embeds = batch["t5_prompt_embeds"].to(dtype=weight_dtype)
                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(dtype=weight_dtype)
                hpcs_lambda = compute_hpcs_lambda(
                    global_step=global_step,
                    total_steps=args.max_train_steps,
                    anchor_ratio=args.hpcs_anchor_ratio,
                    eta=args.hpcs_eta,
                    power=args.hpcs_power,
                )
                prompt_embeds = hpcs_conditioner(
                    clip_hidden_states=clip_prompt_embeds,
                    t5_hidden_states=t5_prompt_embeds,
                    context_weight=hpcs_lambda,
                ).to(dtype=weight_dtype)

                # The paper samples exactly one control from {Mask, Width, Sketch, Edge, Partial}.
                controlnet_images = batch["conditioning_pixel_values"].to(dtype=vae.dtype)
                if controlnet_images.ndim == 4:
                    controlnet_images = controlnet_images.unsqueeze(1)
                num_conditioning_images = controlnet_images.shape[1]
                if num_conditioning_images != 1:
                    raise ValueError(
                        f"Paper-aligned training expects exactly one control image, got {num_conditioning_images}."
                    )
                flat_controlnet_images = controlnet_images.reshape(
                    controlnet_images.shape[0] * num_conditioning_images, *controlnet_images.shape[2:]
                )
                flat_controlnet_images = vae.encode(flat_controlnet_images).latent_dist.sample()
                flat_controlnet_images = (
                    flat_controlnet_images - vae.config.shift_factor
                ) * vae.config.scaling_factor
                controlnet_images = flat_controlnet_images.view(
                    bsz, num_conditioning_images, *flat_controlnet_images.shape[1:]
                ).to(dtype=weight_dtype)
                control_block_res_samples = controlnet(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    controlnet_cond=controlnet_images[:, 0],
                    return_dict=False,
                )[0]
                control_block_res_samples = [
                    sample.to(dtype=weight_dtype) for sample in control_block_res_samples
                ]

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    block_controlnet_hidden_states=control_block_res_samples,
                    return_dict=False,
                )[0]

                # Paper Eqs. (3), (5), and (6): v = z_1 - z_0 and exact SLRM weighting.
                target = velocity_target(model_input, noise)
                equivalent_diameter_px = batch["equivalent_diameter_px"].to(model_input.device)
                loss, per_sample_loss, slrm_weights = weighted_velocity_mse(
                    prediction=model_pred,
                    target=target,
                    diameter=equivalent_diameter_px,
                    gamma=args.slrm_gamma,
                    dmax=args.slrm_dmax,
                    epsilon=args.slrm_epsilon,
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = list(controlnet.parameters()) + list(hpcs_conditioner.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process and args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint-")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[:num_to_remove]
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                            for removing_checkpoint in removing_checkpoints:
                                shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint))
                    accelerator.wait_for_everyone()
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    # Every rank saves its RNG state; the model hook writes shared weights on rank zero.
                    accelerator.save_state(save_path)
                    if accelerator.is_main_process:
                        logger.info(f"Saved state to {save_path}")

                if (
                    accelerator.is_main_process
                    and args.validation_prompt is not None
                    and global_step % args.validation_steps == 0
                ):
                    image_logs = log_validation(
                        controlnet,
                        hpcs_conditioner,
                        transformer,
                        vae,
                        args,
                        accelerator,
                        weight_dtype,
                        global_step,
                    )

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "hpcs_lambda": hpcs_lambda,
                "flow_t_mean": flow_t.detach().float().mean().item(),
                "slrm_w_mean": slrm_weights.detach().mean().item(),
                "slrm_w_min": slrm_weights.detach().min().item(),
                "slrm_w_max": slrm_weights.detach().max().item(),
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = unwrap_model(controlnet)
        hpcs_conditioner = unwrap_model(hpcs_conditioner)
        controlnet.save_pretrained(os.path.join(args.output_dir, "controlnet"))
        hpcs_conditioner.save_pretrained(os.path.join(args.output_dir, "hpcs"))

        # Run a final round of validation.
        image_logs = None
        if args.validation_prompt is not None:
            image_logs = log_validation(
                controlnet=controlnet,
                hpcs_conditioner=hpcs_conditioner,
                transformer=transformer,
                vae=vae,
                args=args,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                step=global_step,
                is_final_validation=True,
            )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
