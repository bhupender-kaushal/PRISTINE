from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn


HPCS_CONFIG_NAME = "config.json"
HPCS_WEIGHT_NAME = "model.safetensors"


def compute_hpcs_lambda(
    global_step: int,
    total_steps: int,
    anchor_ratio: float = 0.4,
    eta: float = 0.4,
    power: float = 1.0,
) -> float:
    """Compute Eq. (4)'s progressive T5 injection weight.

    The paper states
        lambda(tau) = clip((tau - tau_anchor) / (tau_total * eta), 0, 1)^k.
    Ratios are used here so the schedule remains invariant to the total run length.
    """

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= anchor_ratio <= 1.0:
        raise ValueError("anchor_ratio must be in [0, 1]")
    if eta <= 0.0:
        raise ValueError("eta must be positive")
    if power <= 0.0:
        raise ValueError("power must be positive")

    progress = float(global_step) / float(total_steps)
    ramp = max(0.0, min(1.0, (progress - anchor_ratio) / eta))
    return float(ramp**power)


class HPCSConditioner(nn.Module):
    """Literal additive implementation of CSGen Eq. (4).

    CLIP-L/G token features form the geometric branch and T5 token features form
    the contextual branch. The shorter sequence is zero-padded so learned Phi
    and Psi projections can be added token-wise:

        E_unified = Phi(h_geo) + lambda * Psi(h_ctx)

    The projections are saved beside ControlNet and are treated as part of the
    trainable ControlNet conditioning branch.
    """

    def __init__(
        self,
        clip_dim: int = 2048,
        t5_dim: int = 4096,
        joint_dim: int = 4096,
    ) -> None:
        super().__init__()
        self.clip_dim = int(clip_dim)
        self.t5_dim = int(t5_dim)
        self.joint_dim = int(joint_dim)
        self.phi = nn.Linear(self.clip_dim, self.joint_dim, bias=False)
        self.psi = nn.Linear(self.t5_dim, self.joint_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Identity-like initialization preserves SD3's usual CLIP zero-padding
        # and T5 representation before the learned projections adapt.
        with torch.no_grad():
            self.phi.weight.zero_()
            self.psi.weight.zero_()
            self.phi.weight.diagonal().fill_(1.0)
            self.psi.weight.diagonal().fill_(1.0)

    @staticmethod
    def _pad_tokens(hidden_states: torch.Tensor, target_length: int) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected [batch, tokens, channels], got {tuple(hidden_states.shape)}")
        token_count = hidden_states.shape[1]
        if token_count == target_length:
            return hidden_states
        if token_count > target_length:
            return hidden_states[:, :target_length]
        padding = hidden_states.new_zeros(
            hidden_states.shape[0], target_length - token_count, hidden_states.shape[2]
        )
        return torch.cat([hidden_states, padding], dim=1)

    def forward(
        self,
        clip_hidden_states: torch.Tensor,
        t5_hidden_states: torch.Tensor,
        context_weight: float | torch.Tensor,
    ) -> torch.Tensor:
        if clip_hidden_states.shape[0] != t5_hidden_states.shape[0]:
            raise ValueError("CLIP and T5 batch sizes must match")
        if clip_hidden_states.shape[-1] != self.clip_dim:
            raise ValueError(
                f"Expected CLIP width {self.clip_dim}, got {clip_hidden_states.shape[-1]}"
            )
        if t5_hidden_states.shape[-1] != self.t5_dim:
            raise ValueError(f"Expected T5 width {self.t5_dim}, got {t5_hidden_states.shape[-1]}")

        target_length = max(clip_hidden_states.shape[1], t5_hidden_states.shape[1])
        clip_hidden_states = self._pad_tokens(clip_hidden_states, target_length)
        t5_hidden_states = self._pad_tokens(t5_hidden_states, target_length)

        geometric = self.phi(clip_hidden_states)
        contextual = self.psi(t5_hidden_states)
        weight = torch.as_tensor(
            context_weight,
            device=contextual.device,
            dtype=contextual.dtype,
        )
        if weight.ndim == 0:
            weight = weight.reshape(1, 1, 1)
        elif weight.ndim == 1:
            weight = weight.reshape(-1, 1, 1)
        return geometric + weight * contextual

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "clip_dim": self.clip_dim,
            "t5_dim": self.t5_dim,
            "joint_dim": self.joint_dim,
            "fusion": "additive_eq4",
        }
        (output_dir / HPCS_CONFIG_NAME).write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        state = {key: value.detach().cpu().contiguous() for key, value in self.state_dict().items()}
        save_file(state, str(output_dir / HPCS_WEIGHT_NAME))

    @classmethod
    def from_pretrained(
        cls,
        input_dir: str | Path,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "HPCSConditioner":
        input_dir = Path(input_dir)
        config = json.loads((input_dir / HPCS_CONFIG_NAME).read_text(encoding="utf-8"))
        model = cls(
            clip_dim=int(config["clip_dim"]),
            t5_dim=int(config["t5_dim"]),
            joint_dim=int(config["joint_dim"]),
        )
        state = load_file(str(input_dir / HPCS_WEIGHT_NAME), device="cpu")
        model.load_state_dict(state)
        if device is not None or dtype is not None:
            model.to(device=device, dtype=dtype)
        return model
