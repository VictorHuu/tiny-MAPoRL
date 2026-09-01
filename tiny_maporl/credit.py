from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CreditBatch:
    agent_a: torch.Tensor
    agent_b: torch.Tensor


class SharedTeamCredit:
    name = "shared_team"

    def __call__(self, team_rewards: torch.Tensor) -> CreditBatch:
        return CreditBatch(
            agent_a=team_rewards.clone(),
            agent_b=team_rewards.clone(),
        )


def group_relative_advantage(
    credits: torch.Tensor,
    group_size: int,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, float]:
    grouped = credits.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, unbiased=False, keepdim=True)
    advantages = (grouped - mean) / (std + eps)
    zero_std_fraction = (std.squeeze(1) < eps).float().mean().item()
    return advantages.reshape(-1), zero_std_fraction
