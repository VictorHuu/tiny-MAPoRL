from __future__ import annotations

import re
from typing import Any

_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def extract_gsm8k_ground_truth(answer: str) -> str:
    """Extract the canonical final answer from a GSM8K answer string."""
    if "####" in answer:
        answer = answer.rsplit("####", 1)[-1]
    return normalize_answer(answer)


def _completion_to_text(completion: Any) -> str:
    """Accept both standard and conversational TRL completion formats."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        # Conversational completions are commonly represented as a list of messages.
        for item in reversed(completion):
            if isinstance(item, dict) and item.get("role") == "assistant":
                return str(item.get("content", ""))
        return " ".join(_completion_to_text(item) for item in completion)
    return str(completion)


def normalize_answer(text: str) -> str:
    """Normalize a short numeric answer for exact-match grading."""
    text = text.strip().replace(",", "").replace("$", "")
    if text.endswith("."):
        text = text[:-1]
    try:
        value = float(text)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except ValueError:
        return text.strip().lower()


def extract_final_answer(completion: Any) -> str:
    """Extract the final boxed answer, falling back to the last numeric token."""
    text = _completion_to_text(completion)
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return normalize_answer(boxed[-1])
    numbers = _NUMBER_RE.findall(text.replace(",", ""))
    if numbers:
        return normalize_answer(numbers[-1])
    return normalize_answer(text)


def exact_answer_reward(completions, ground_truth, log_extra=None, log_metric=None, **kwargs):
    """Binary verifier reward for GSM8K-style tasks."""
    predictions = [extract_final_answer(c) for c in completions]
    targets = [normalize_answer(str(x)) for x in ground_truth]
    rewards = [1.0 if pred == target else 0.0 for pred, target in zip(predictions, targets)]

    if log_extra:
        log_extra("predicted_answer", predictions)
        log_extra("ground_truth", targets)
    if log_metric and rewards:
        log_metric("exact_accuracy", sum(rewards) / len(rewards))
    return rewards


def consultation_reward(environments=None, **kwargs):
    """Small shaping reward for actually consulting the peer agent once.

    This is deliberately tiny relative to the task reward. It prevents the active
    policy from learning to ignore the multi-agent interaction entirely while
    leaving task correctness as the dominant objective.
    """
    if environments is None:
        return []
    return [1.0 if getattr(env, "consult_count", 0) == 1 else 0.0 for env in environments]
