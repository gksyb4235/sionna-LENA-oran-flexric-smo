"""Measured next-interval rewards; no surrogate forecasts or future labels."""
import numpy as np


def reward(app, variant, previous, current, movement=0.0):
    if variant not in ("local", "guarded"):
        raise ValueError(variant)
    p, c = previous["short"], current["short"]
    if app == "lb":
        objective = (2 * (c["jain"] - 0.7) + 2 * (c["jain"] - p["jain"])) if c["active_load"] else 0
    elif app == "mo":
        # Zero HO receives neither a perfect ratio nor an improvement bonus.
        # Successful HO support prevents failed attempts from creating apparent
        # evidence of a good ping-pong/attempt ratio.
        confidence = min(p["ho_success_support"], c["ho_success_support"])
        objective = (5 * c["ho_success_support"] * (0.1 - c["pp_posterior"])
                     + 2 * confidence * (p["pp_posterior"] - c["pp_posterior"]))
    else:
        raise ValueError(app)
    deficit = max(0, 0.95 - c["delivery_ratio"]) if c["has_traffic"] else 0
    # Both variants have minimum anti-reward-hacking safeguards. "local" is not
    # a deliberately unsafe app. "guarded" additionally values the other app.
    rlf_penalty = (0.8 if variant == "guarded" else 0.15) * c["rlf_per_1000_ue_s"]
    delivery_penalty = (4.0 if variant == "guarded" else 0.5) * deficit
    # Failures may complete after their attempt's observation window.
    failure_penalty = (2.0 if variant == "guarded" else 0.5) * c["ho_failures"] / (c["ho_attempts"] + 10.0)
    cross_penalty = 0.0
    if variant == "guarded":
        if app == "lb":
            cross_penalty = 3 * c["ho_confidence"] * max(0, c["pp_posterior"] - 0.1)
        else:
            cross_penalty = 2 * max(0, 0.9 - c["jain"]) if c["active_load"] else 0
    change_penalty = 0.02 * movement
    raw = objective - rlf_penalty - delivery_penalty - failure_penalty - cross_penalty - change_penalty
    return float(np.clip(raw, -5, 5)), {"objective": float(objective),
        "rlf_penalty": float(rlf_penalty), "delivery_penalty": float(delivery_penalty),
        "ho_failure_penalty": float(failure_penalty),
        "cross_objective_penalty": float(cross_penalty), "change_penalty": float(change_penalty),
        "unclipped_reward": float(raw), "variant": variant,
        "pp_ratio_is_estimated": c["ho_attempts"] < 10,
        "ho_attempts": c["ho_attempts"], "pingpongs": c["pingpongs"]}
