from __future__ import annotations

from collections.abc import Sequence

import torch


def _as_prompt_list(prompt: str | Sequence[str]) -> list[str]:
    if isinstance(prompt, str):
        return [prompt]
    return list(prompt)


def encode_t5_prompt(
    text_encoder,
    tokenizer,
    prompt: str | Sequence[str],
    *,
    max_sequence_length: int,
    device: torch.device | str,
    num_images_per_prompt: int = 1,
) -> torch.Tensor:
    prompts = _as_prompt_list(prompt)
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    batch_size, sequence_length, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    return prompt_embeds.view(batch_size * num_images_per_prompt, sequence_length, -1)


def encode_clip_prompt(
    text_encoder,
    tokenizer,
    prompt: str | Sequence[str],
    *,
    device: torch.device | str,
    num_images_per_prompt: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompts = _as_prompt_list(prompt)
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    encoder_output = text_encoder(
        text_inputs.input_ids.to(device),
        output_hidden_states=True,
    )
    pooled_prompt_embeds = encoder_output[0]
    prompt_embeds = encoder_output.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    batch_size, sequence_length, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, sequence_length, -1)
    pooled_prompt_embeds = pooled_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
    return prompt_embeds, pooled_prompt_embeds


def encode_prompt_components(
    text_encoders,
    tokenizers,
    *,
    clip_prompt: str | Sequence[str],
    t5_prompt: str | Sequence[str],
    max_sequence_length: int,
    device: torch.device | str | None = None,
    num_images_per_prompt: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode CSGen's rule and contextual prompts without standard SD3 concatenation.

    The two CLIP penultimate hidden states are concatenated channel-wise to form
    h_geo. T5 forms h_ctx. HPCSConditioner applies the learned projections and
    additive progressive fusion described by Eq. (4).
    """

    if len(text_encoders) != 3 or len(tokenizers) != 3:
        raise ValueError("CSGen requires two CLIP encoders/tokenizers and one T5 encoder/tokenizer.")
    clip_prompts = _as_prompt_list(clip_prompt)
    t5_prompts = _as_prompt_list(t5_prompt)
    if len(clip_prompts) != len(t5_prompts):
        raise ValueError("CLIP rule prompts and T5 VLM prompts must have matching batch sizes.")

    clip_hidden_states = []
    pooled_hidden_states = []
    for tokenizer, text_encoder in zip(tokenizers[:2], text_encoders[:2]):
        encoder_device = device if device is not None else text_encoder.device
        hidden_states, pooled = encode_clip_prompt(
            text_encoder,
            tokenizer,
            clip_prompts,
            device=encoder_device,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_hidden_states.append(hidden_states)
        pooled_hidden_states.append(pooled)

    t5_device = device if device is not None else text_encoders[2].device
    t5_hidden_states = encode_t5_prompt(
        text_encoders[2],
        tokenizers[2],
        t5_prompts,
        max_sequence_length=max_sequence_length,
        device=t5_device,
        num_images_per_prompt=num_images_per_prompt,
    )
    return (
        torch.cat(clip_hidden_states, dim=-1),
        torch.cat(pooled_hidden_states, dim=-1),
        t5_hidden_states,
    )
