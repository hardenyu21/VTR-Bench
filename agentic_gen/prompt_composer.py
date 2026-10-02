from __future__ import annotations

import hashlib


ORIGINAL_HEADER = "[Original video request — authoritative; preserve exactly]"
MOTION_HEADER = "[Additional motion guidance — additive only]"
PRECEDENCE_RULE = (
    "The original video request is authoritative. If the additional motion guidance conflicts "
    "with it, follow the original request. Preserve every required visible text string exactly."
)


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def deduplicate_motion_refinement(original_prompt: str, motion_refinement: str) -> str:
    """Remove only a verbatim copied original prompt from a refinement.

    Semantic rewriting is deliberately avoided: it could drop required text. The planner contract
    requires a delta-only refinement, while this exact-prefix removal protects against a provider
    redundantly copying the immutable original prompt.
    """

    original = original_prompt.strip()
    refinement = motion_refinement.strip()
    if not original:
        raise ValueError("original_prompt must not be empty")
    if not refinement:
        raise ValueError("motion_refinement must not be empty")
    if refinement.startswith(original):
        refinement = refinement[len(original) :].lstrip(" \t\r\n:-")
    if not refinement:
        raise ValueError("motion_refinement must add guidance beyond the original prompt")
    return refinement


def compose_h3_prompt(original_prompt: str, motion_refinement: str) -> tuple[str, str]:
    """Return the deterministic H3 prompt and normalized delta-only refinement."""

    refinement = deduplicate_motion_refinement(original_prompt, motion_refinement)
    composed = (
        f"{ORIGINAL_HEADER}\n"
        f"{original_prompt}\n\n"
        f"{MOTION_HEADER}\n"
        f"{refinement}\n\n"
        f"{PRECEDENCE_RULE}"
    )
    return composed, refinement
