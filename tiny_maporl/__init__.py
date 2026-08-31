"""Minimal multi-agent post-training components built on upstream TRL."""

from .data import load_gsm8k_dataset
from .environment import PeerReviewEnv
from .rewards import exact_answer_reward, consultation_reward, extract_final_answer, extract_gsm8k_ground_truth

__all__ = [
    "PeerReviewEnv",
    "consultation_reward",
    "exact_answer_reward",
    "extract_final_answer",
    "extract_gsm8k_ground_truth",
    "load_gsm8k_dataset",
]
