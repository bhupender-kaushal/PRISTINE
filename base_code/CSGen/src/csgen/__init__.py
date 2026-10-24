"""CSGen conditioning and training utilities for Diffusers."""

from .models import HPCSConditioner, compute_hpcs_lambda
from .utils.flow import flow_interpolate, sample_uniform_t, scheduler_timestep, velocity_target
from .utils.prompts import ensure_domain_prefix, prompt_domain
from .utils.slrm import equivalent_diameter, slrm_weight, weighted_velocity_mse

__version__ = "0.1.0"

__all__ = [
    "HPCSConditioner",
    "compute_hpcs_lambda",
    "ensure_domain_prefix",
    "equivalent_diameter",
    "flow_interpolate",
    "prompt_domain",
    "sample_uniform_t",
    "scheduler_timestep",
    "slrm_weight",
    "velocity_target",
    "weighted_velocity_mse",
]
