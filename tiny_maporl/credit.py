from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CreditBatch:
    agent_a_draft: torch.Tensor
    agent_b: torch.Tensor
    agent_a_final: torch.Tensor


class SharedTeamCredit:
    name = "shared_team"

    def __call__(self, team_rewards: torch.Tensor, **kwargs) -> CreditBatch:
        return CreditBatch(
            agent_a_draft=team_rewards.clone(),
            agent_b=team_rewards.clone(),
            agent_a_final=team_rewards.clone(),
        )


class DiscountedInfluenceCredit:
    name = "discounted_influence"

    def __init__(self, discount: float = 0.3) -> None:
        self.discount = discount

    def __call__(
        self,
        team_rewards: torch.Tensor,
        draft_correct: torch.Tensor,
        agent_b_correct: torch.Tensor,
        **kwargs,
    ) -> CreditBatch:
        gamma = self.discount
        agent_a_draft = (
            draft_correct + gamma * agent_b_correct + gamma**2 * team_rewards
        ) / (1.0 + gamma + gamma**2)
        agent_b = (agent_b_correct + gamma * team_rewards) / (1.0 + gamma)
        return CreditBatch(
            agent_a_draft=agent_a_draft,
            agent_b=agent_b,
            agent_a_final=team_rewards.clone(),
        )


def group_relative_advantage(
    credits: torch.Tensor,
    group_size: int,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, float]:
    grouped = credits.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    if group_size > 1:
        std = grouped.std(dim=1, keepdim=True)
    else:
        std = torch.zeros_like(mean)
    advantages = (grouped - mean) / (std + eps)
    zero_std_fraction = (std.squeeze(1) < eps).float().mean().item()
    return advantages.reshape(-1), zero_std_fraction
