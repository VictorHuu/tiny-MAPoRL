from __future__ import annotations

from .partner import FrozenPartner


class PeerReviewEnv:
    """Stateful TRL environment exposing one frozen LLM peer as a tool."""

    def __init__(self, partner: FrozenPartner) -> None:
        self.partner = partner
        self.question = ""
        self.ground_truth = ""
        self.consult_count = 0
        self.last_draft = ""
        self.last_feedback = ""

    def reset(self, question: str, ground_truth: str, **kwargs) -> None:
        """Reset per-rollout state; the dataset already provides the user prompt."""
        self.question = question
        self.ground_truth = str(ground_truth)
        self.consult_count = 0
        self.last_draft = ""
        self.last_feedback = ""

    def ask_partner(self, draft: str) -> str:
        """Ask the peer agent to review a draft solution.

        Args:
            draft: The active agent's current proposed reasoning and answer.

        Returns:
            The frozen peer agent's concise critique or confirmation.
        """
        self.consult_count += 1
        self.last_draft = draft
        self.last_feedback = self.partner.review(self.question, draft)
        return self.last_feedback
