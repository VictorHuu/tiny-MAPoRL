from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_TRAJECTORY_LOG_PATH: Path | None = None
_TRAJECTORY_LOG_LOCK = threading.Lock()


def configure_trajectory_log(path: str) -> None:
    global _TRAJECTORY_LOG_PATH
    _TRAJECTORY_LOG_PATH = Path(path)
    _TRAJECTORY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


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


def exact_answer_reward(
    completions,
    ground_truth,
    environments=None,
    trainer_state=None,
    log_extra=None,
    log_metric=None,
    **kwargs,
):
    predictions = [extract_final_answer(c) for c in completions]
    targets = [normalize_answer(str(x)) for x in ground_truth]
    rewards = [1.0 if pred == target else 0.0 for pred, target in zip(predictions, targets)]
    envs = environments or [None] * len(completions)

    drafts = [getattr(env, "last_draft", "") if env is not None else "" for env in envs]
    feedbacks = [getattr(env, "last_feedback", "") if env is not None else "" for env in envs]
    consult_counts = [getattr(env, "consult_count", 0) if env is not None else 0 for env in envs]
    draft_predictions = [extract_final_answer(draft) if draft else "" for draft in drafts]
    draft_correct = [draft == target for draft, target in zip(draft_predictions, targets)]

    transitions = []
    for consulted, before, after in zip(consult_counts, draft_correct, rewards):
        if consulted != 1:
            transitions.append("no_consult")
        elif not before and after == 1.0:
            transitions.append("WR_correction")
        elif before and after == 1.0:
            transitions.append("RR_preservation")
        elif before and after == 0.0:
            transitions.append("RW_corruption")
        else:
            transitions.append("WW_failure")

    if log_extra:
        log_extra("draft", drafts)
        log_extra("partner_feedback", feedbacks)
        log_extra("draft_answer", draft_predictions)
        log_extra("predicted_answer", predictions)
        log_extra("ground_truth", targets)
        log_extra("consult_count", consult_counts)
        log_extra("interaction_transition", transitions)

    if log_metric and rewards:
        log_metric("exact_accuracy", sum(rewards) / len(rewards))
        log_metric("draft_accuracy", sum(draft_correct) / len(draft_correct))
        log_metric("correction_rate", transitions.count("WR_correction") / len(transitions))
        log_metric("preservation_rate", transitions.count("RR_preservation") / len(transitions))
        log_metric("corruption_rate", transitions.count("RW_corruption") / len(transitions))
        log_metric("failure_rate", transitions.count("WW_failure") / len(transitions))
        log_metric("consult_once_rate", sum(x == 1 for x in consult_counts) / len(consult_counts))

    if _TRAJECTORY_LOG_PATH is not None:
        step = int(getattr(trainer_state, "global_step", -1))
        rows = []
        for i, completion in enumerate(completions):
            rows.append(
                {
                    "step": step,
                    "question": getattr(envs[i], "question", "") if envs[i] is not None else "",
                    "draft": drafts[i],
                    "partner_feedback": feedbacks[i],
                    "final": _completion_to_text(completion),
                    "draft_answer": draft_predictions[i],
                    "final_answer": predictions[i],
                    "ground_truth": targets[i],
                    "consult_count": consult_counts[i],
                    "transition": transitions[i],
                    "task_reward": rewards[i],
                }
            )
        with _TRAJECTORY_LOG_LOCK:
            with _TRAJECTORY_LOG_PATH.open("a", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return rewards


def consultation_reward(completions=None, environments=None, **kwargs):
    """Small shaping reward for actually consulting the peer agent once.

    This is deliberately tiny relative to the task reward. It prevents the active
    policy from learning to ignore the multi-agent interaction entirely while
    leaving task correctness as the dominant objective.
    """
    if environments is None:
        return [0.0] * len(completions or [])
    return [1.0 if getattr(env, "consult_count", 0) == 1 else 0.0 for env in environments]
