#!/usr/bin/env python
# coding=utf-8

import logging
import random
from collections import Counter, defaultdict

import numpy as np
from datasets import Dataset


def _coerce_dict_value(value) -> dict:
    return value if isinstance(value, dict) else {}


def _normalize_bucket(value, fallback: str = "unknown") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _orientation_bucket_from_deg(value) -> str:
    try:
        deg = float(value) % 180.0
    except (TypeError, ValueError):
        return "unknown"

    if deg < 22.5 or deg >= 157.5:
        return "horizontal"
    if deg < 67.5:
        return "diagonal"
    if deg < 112.5:
        return "vertical"
    return "diagonal"


def _build_counterfactual_signature(example, relaxed: bool = False) -> tuple[str, ...]:
    attribute_bins = _coerce_dict_value(example.get("attribute_bins", {}))
    scale_meta = _coerce_dict_value(example.get("scale_meta", {}))
    geometry_raw = _coerce_dict_value(example.get("geometry_raw", {}))

    structural_parts = [
        _normalize_bucket(attribute_bins.get("width_desc")),
        _normalize_bucket(attribute_bins.get("curvature_desc")),
        _normalize_bucket(attribute_bins.get("branch_desc")),
        _normalize_bucket(attribute_bins.get("continuity_desc")),
    ]
    if relaxed:
        return tuple(structural_parts)

    orientation_desc = attribute_bins.get("orientation_desc")
    if orientation_desc is None:
        orientation_desc = _orientation_bucket_from_deg(geometry_raw.get("orientation_deg"))

    return tuple(
        structural_parts
        + [
            _normalize_bucket(scale_meta.get("scale_bin")),
            _normalize_bucket(orientation_desc),
        ]
    )


def _sample_domain(domains: list[str], probabilities: list[float], rng: random.Random) -> str:
    return rng.choices(domains, weights=probabilities, k=1)[0]


def _sample_partner_index(
    anchor_index: int,
    exact_partner_pools: list[list[int]],
    relaxed_partner_pools: list[list[int]],
    rng: random.Random,
) -> tuple[int | None, str | None]:
    exact_candidates = exact_partner_pools[anchor_index]
    if exact_candidates:
        return rng.choice(exact_candidates), "exact"

    relaxed_candidates = relaxed_partner_pools[anchor_index]
    if relaxed_candidates:
        return rng.choice(relaxed_candidates), "relaxed"

    return None, None


def prepare_local_processed_training_view(train_split: Dataset, args, logger: logging.Logger | None = None) -> tuple[Dataset, dict]:
    logger = logger or logging.getLogger(__name__)

    num_samples = len(train_split)
    if num_samples == 0:
        return train_split, {"strategy": "empty"}

    enable_domain_rebalance = bool(getattr(args, "enable_domain_rebalance", False))
    domain_balance_alpha = float(getattr(args, "domain_balance_alpha", 1.0))
    counterfactual_pair_prob = float(getattr(args, "counterfactual_pair_prob", 0.0))
    target_size = int(getattr(args, "max_train_samples", None) or num_samples)

    if not enable_domain_rebalance and counterfactual_pair_prob <= 0.0 and target_size >= num_samples:
        return train_split, {
            "strategy": "identity",
            "target_size": num_samples,
            "original_domain_counts": dict(Counter(str(example.get("domain", "unknown")) for example in train_split)),
        }

    seed_value = getattr(args, "seed", None)
    rng = random.Random(0 if seed_value is None else int(seed_value))

    domains_per_index: list[str] = []
    domain_to_indices: dict[str, list[int]] = defaultdict(list)
    exact_groups: dict[tuple[str, ...], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    relaxed_groups: dict[tuple[str, ...], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    exact_keys: list[tuple[str, ...]] = []
    relaxed_keys: list[tuple[str, ...]] = []

    for idx, example in enumerate(train_split):
        domain = str(example.get("domain", example.get("class_name", "unknown")))
        domains_per_index.append(domain)
        domain_to_indices[domain].append(idx)

        exact_key = _build_counterfactual_signature(example, relaxed=False)
        relaxed_key = _build_counterfactual_signature(example, relaxed=True)
        exact_keys.append(exact_key)
        relaxed_keys.append(relaxed_key)
        exact_groups[exact_key][domain].append(idx)
        relaxed_groups[relaxed_key][domain].append(idx)

    sorted_domains = sorted(domain_to_indices)
    domain_counts = {domain: len(indices) for domain, indices in domain_to_indices.items()}
    if enable_domain_rebalance:
        domain_probabilities = np.asarray(
            [float(domain_counts[domain]) ** domain_balance_alpha for domain in sorted_domains],
            dtype=np.float64,
        )
    else:
        domain_probabilities = np.asarray([float(domain_counts[domain]) for domain in sorted_domains], dtype=np.float64)
    domain_probabilities = (domain_probabilities / max(domain_probabilities.sum(), 1e-8)).tolist()

    exact_partner_pools: list[list[int]] = []
    relaxed_partner_pools: list[list[int]] = []
    for idx, domain in enumerate(domains_per_index):
        exact_partner_pools.append(
            [
                candidate
                for candidate_domain, candidate_indices in exact_groups[exact_keys[idx]].items()
                if candidate_domain != domain
                for candidate in candidate_indices
            ]
        )
        relaxed_partner_pools.append(
            [
                candidate
                for candidate_domain, candidate_indices in relaxed_groups[relaxed_keys[idx]].items()
                if candidate_domain != domain
                for candidate in candidate_indices
            ]
        )

    selected_indices: list[int] = []
    counterfactual_exact = 0
    counterfactual_relaxed = 0
    counterfactual_missed = 0

    while len(selected_indices) < target_size:
        sampled_domain = _sample_domain(sorted_domains, domain_probabilities, rng)
        anchor_index = rng.choice(domain_to_indices[sampled_domain])
        selected_indices.append(anchor_index)

        if len(selected_indices) >= target_size or counterfactual_pair_prob <= 0.0:
            continue
        if rng.random() > counterfactual_pair_prob:
            continue

        partner_index, partner_mode = _sample_partner_index(
            anchor_index,
            exact_partner_pools=exact_partner_pools,
            relaxed_partner_pools=relaxed_partner_pools,
            rng=rng,
        )
        if partner_index is None:
            counterfactual_missed += 1
            continue

        selected_indices.append(partner_index)
        if partner_mode == "exact":
            counterfactual_exact += 1
        elif partner_mode == "relaxed":
            counterfactual_relaxed += 1

    selected_indices = selected_indices[:target_size]
    selected_split = train_split.select(selected_indices)

    sampled_domain_counts = Counter(domains_per_index[idx] for idx in selected_indices)
    stats = {
        "strategy": "domain_rebalanced" if enable_domain_rebalance else "resampled",
        "target_size": target_size,
        "domain_balance_alpha": domain_balance_alpha,
        "counterfactual_pair_prob": counterfactual_pair_prob,
        "original_domain_counts": dict(sorted(domain_counts.items())),
        "sampled_domain_counts": dict(sorted(sampled_domain_counts.items())),
        "counterfactual_exact_pairs": counterfactual_exact,
        "counterfactual_relaxed_pairs": counterfactual_relaxed,
        "counterfactual_pair_misses": counterfactual_missed,
        "selected_unique_samples": len(set(selected_indices)),
    }

    logger.info("Prepared local processed training view: %s", stats)
    return selected_split, stats
