from __future__ import annotations

import re


DOMAIN_PREFIX_PATTERN = re.compile(r"^\s*\[DOMAIN=([^\]]+)\]")


def prompt_domain(prompt: str) -> str | None:
    match = DOMAIN_PREFIX_PATTERN.match(prompt)
    return match.group(1).strip() if match else None


def ensure_domain_prefix(prompt: str, domain: str) -> str:
    """Add the explicit paper-required domain prefix and reject contradictions."""

    prompt = str(prompt).strip()
    domain = str(domain).strip()
    if not prompt:
        return ""
    if not domain or "]" in domain:
        raise ValueError(f"Invalid domain value for prompt prefix: {domain!r}")
    existing_domain = prompt_domain(prompt)
    if existing_domain is not None:
        if existing_domain != domain:
            raise ValueError(
                f"Prompt domain {existing_domain!r} conflicts with sample domain {domain!r}."
            )
        return prompt
    return f"[DOMAIN={domain}] {prompt}"
