from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig

from .credit import SharedTeamCredit, group_relative_advantage
from .rewards import extract_final_answer, normalize_answer


@dataclass
class ActionTrace:
    prompt_ids: list[int]
    response_ids: list[int]
    old_logps: torch.Tensor
    ref_logps: torch.Tensor


@dataclass
class JointTrajectory:
    question: str
    ground_truth: str
    agent_a_draft: str
    agent_b_message: str
    agent_a_final: str
    action_a_draft: ActionTrace
    action_b: ActionTrace
    action_a_final: ActionTrace
    team_reward: float
    credit_a: float = 0.0
    credit_b: float = 0.0
    advantage_a: float = 0.0
    advantage_b: float = 0.0


class JointGRPOTrainer:
    def __init__(
        self,
        model_a_path: str,
        model_b_path: str,
        train_dataset: Dataset,
        args: GRPOConfig,
        device_a: str = "cuda:0",
        device_b: str = "cuda:0",
        lora_r: int = 16,
        lora_alpha: int = 32,
        load_in_4bit: bool = True,
        credit_assigner=None,
        report_to: str = "none",
        save_steps: int = 50,
        seed: int = 42,
    ) -> None:
        self.args = args
        self.dataset = train_dataset
        self.device_a = torch.device(device_a)
        self.device_b = torch.device(device_b)
        self.credit_assigner = credit_assigner or SharedTeamCredit()
        self.report_to = report_to
        self.save_steps = save_steps
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_path = self.output_dir / "joint_trajectories.jsonl"
        self.metrics_path = self.output_dir / "joint_metrics.jsonl"
        self.seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.tokenizer_a, self.model_a = self._load_agent(
            model_a_path, self.device_a, lora_r, lora_alpha, load_in_4bit
        )
        self.tokenizer_b, self.model_b = self._load_agent(
            model_b_path, self.device_b, lora_r, lora_alpha, load_in_4bit
        )

        self.params_a = [p for p in self.model_a.parameters() if p.requires_grad]
        self.params_b = [p for p in self.model_b.parameters() if p.requires_grad]
        self.optimizer_a = torch.optim.AdamW(self.params_a, lr=args.learning_rate)
        self.optimizer_b = torch.optim.AdamW(self.params_b, lr=args.learning_rate)

        self.wandb = None
        if report_to == "wandb":
            import wandb

            self.wandb = wandb
            wandb.init(
                project="tiny-MAPoRL",
                name=self.output_dir.name,
                config={
                    "model_a": model_a_path,
                    "model_b": model_b_path,
                    "num_generations": args.num_generations,
                    "num_iterations": args.num_iterations,
                    "beta": args.beta,
                    "epsilon": args.epsilon,
                    "credit": self.credit_assigner.name,
                },
            )

    def _load_agent(
        self,
        model_path: str,
        device: torch.device,
        lora_r: int,
        lora_alpha: int,
        load_in_4bit: bool,
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {"dtype": torch.bfloat16}
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = {"": str(device)}

        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if not load_in_4bit:
            model.to(device)
        else:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
        model = get_peft_model(model, peft_config)
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        model.config.use_cache=False
        return tokenizer, model

    def _chat_ids(self, tokenizer, messages, device: torch.device) -> torch.Tensor:
        kwargs = {
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
        try:
            ids = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            ids = tokenizer.apply_chat_template(messages, **kwargs)
        return ids.to(device)

    def _token_logps(
        self,
        model,
        prompt_ids: list[int],
        response_ids: list[int],
    ) -> torch.Tensor:
        device = model.device
        prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        response = torch.tensor(response_ids, dtype=torch.long, device=device).unsqueeze(0)
        full = torch.cat([prompt, response], dim=1)
        attention_mask = torch.ones_like(full)
        output = model(input_ids=full, attention_mask=attention_mask, use_cache=False)
        start = prompt.shape[1] - 1
        end = start + response.shape[1]
        logits = output.logits[:, start:end, :]
        logps = torch.log_softmax(logits.float(), dim=-1)
        return logps.gather(-1, response.unsqueeze(-1)).squeeze(-1).squeeze(0)

    def _snapshot_action(
        self,
        model,
        prompt_ids: list[int],
        response_ids: list[int],
    ) -> ActionTrace:
        with torch.no_grad():
            old_logps = self._token_logps(model, prompt_ids, response_ids).detach().cpu()
            with model.disable_adapter():
                ref_logps = self._token_logps(model, prompt_ids, response_ids).detach().cpu()
        return ActionTrace(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            old_logps=old_logps,
            ref_logps=ref_logps,
        )

    def _generate(self, model, tokenizer, messages) -> tuple[str, ActionTrace]:
        prompt = self._chat_ids(tokenizer, messages, model.device)
        generation_kwargs = {
            "max_new_tokens": self.args.max_completion_length,
            "do_sample": self.args.temperature > 0,
            "temperature": max(float(self.args.temperature), 1e-6),
            "top_p": float(self.args.top_p),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if self.args.top_k is not None and self.args.top_k > 0:
            generation_kwargs["top_k"] = int(self.args.top_k)

        model.eval()
        old_use_cache = model.config.use_cache
        model.config.use_cache = True
        with torch.no_grad():
            output = model.generate(prompt, **generation_kwargs)
        model.config.use_cache = old_use_cache
        response = output[0, prompt.shape[1] :]
        if response.numel() == 0:
            response = torch.tensor([tokenizer.eos_token_id], device=prompt.device)
        text = tokenizer.decode(response, skip_special_tokens=True).strip()
        action = self._snapshot_action(
            model,
            prompt[0].detach().cpu().tolist(),
            response.detach().cpu().tolist(),
        )
        return text, action

    def _rollout(self, question: str, ground_truth: str) -> JointTrajectory:
        draft_messages = [
            {
                "role": "system",
                "content": (
                    "You are Agent A, the initial solver in a two-agent math team. "
                    "Solve the problem carefully and end with your candidate numeric answer in \\boxed{...}."
                ),
            },
            {"role": "user", "content": question},
        ]
        agent_a_draft, action_a_draft = self._generate(
            self.model_a, self.tokenizer_a, draft_messages
        )

        b_messages = [
            {
                "role": "system",
                "content": (
                    "You are Agent B in a two-agent math team. Inspect Agent A's draft, identify any "
                    "reasoning or arithmetic error, and give your own corrected candidate answer. "
                    "End with a numeric answer in \\boxed{...}."
                ),
            },
            {
                "role": "user",
                "content": f"Problem:\n{question}\n\nAgent A draft:\n{agent_a_draft}",
            },
        ]
        agent_b_message, action_b = self._generate(
            self.model_b, self.tokenizer_b, b_messages
        )

        final_messages = [
            {
                "role": "system",
                "content": (
                    "You are Agent A. Reconsider your draft using Agent B's message as evidence. "
                    "Return the best final solution and end with the numeric answer in \\boxed{...}."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Problem:\n{question}\n\nYour earlier draft:\n{agent_a_draft}"
                    f"\n\nAgent B message:\n{agent_b_message}"
                ),
            },
        ]
        agent_a_final, action_a_final = self._generate(
            self.model_a, self.tokenizer_a, final_messages
        )

        final_answer = extract_final_answer(agent_a_final)
        target = normalize_answer(str(ground_truth))
        team_reward = float(final_answer == target)
        return JointTrajectory(
            question=question,
            ground_truth=target,
            agent_a_draft=agent_a_draft,
            agent_b_message=agent_b_message,
            agent_a_final=agent_a_final,
            action_a_draft=action_a_draft,
            action_b=action_b,
            action_a_final=action_a_final,
            team_reward=team_reward,
        )

    def _action_loss(
        self,
        model,
        action: ActionTrace,
        advantage: float,
    ) -> tuple[torch.Tensor, float, float]:
        current_logps = self._token_logps(model, action.prompt_ids, action.response_ids)
        old_logps = action.old_logps.to(current_logps.device)
        ref_logps = action.ref_logps.to(current_logps.device)
        ratio = torch.exp(current_logps - old_logps)
        epsilon_low = float(self.args.epsilon)
        epsilon_high = float(
            self.args.epsilon_high if self.args.epsilon_high is not None else self.args.epsilon
        )
        clipped_ratio = torch.clamp(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high)
        adv = torch.tensor(float(advantage), device=current_logps.device)
        policy_loss = -torch.minimum(ratio * adv, clipped_ratio * adv)
        delta = ref_logps - current_logps
        per_token_kl = torch.exp(delta) - delta - 1.0
        loss = (policy_loss + float(self.args.beta) * per_token_kl).mean()
        clip_fraction = ((ratio < 1.0 - epsilon_low) | (ratio > 1.0 + epsilon_high)).float().mean()
        return loss, per_token_kl.mean().item(), clip_fraction.item()

    def _update_policies(self, trajectories: list[JointTrajectory]) -> dict[str, float]:
        policy_epochs = max(1, int(self.args.num_iterations))
        a_action_count = 2 * len(trajectories)
        b_action_count = len(trajectories)
        totals = {
            "loss_a": 0.0,
            "loss_b": 0.0,
            "kl_a": 0.0,
            "kl_b": 0.0,
            "clip_a": 0.0,
            "clip_b": 0.0,
            "grad_norm_a": 0.0,
            "grad_norm_b": 0.0,
        }

        for _ in range(policy_epochs):
            self.model_a.train()
            self.model_b.train()
            self.optimizer_a.zero_grad(set_to_none=True)
            self.optimizer_b.zero_grad(set_to_none=True)

            epoch_loss_a = 0.0
            epoch_loss_b = 0.0
            epoch_kl_a = 0.0
            epoch_kl_b = 0.0
            epoch_clip_a = 0.0
            epoch_clip_b = 0.0

            for trajectory in trajectories:
                for action in (trajectory.action_a_draft, trajectory.action_a_final):
                    loss, kl, clip = self._action_loss(
                        self.model_a, action, trajectory.advantage_a
                    )
                    (loss / a_action_count).backward()
                    epoch_loss_a += loss.detach().item()
                    epoch_kl_a += kl
                    epoch_clip_a += clip

                loss, kl, clip = self._action_loss(
                    self.model_b, trajectory.action_b, trajectory.advantage_b
                )
                (loss / b_action_count).backward()
                epoch_loss_b += loss.detach().item()
                epoch_kl_b += kl
                epoch_clip_b += clip

            grad_norm_a = torch.nn.utils.clip_grad_norm_(self.params_a, self.args.max_grad_norm)
            grad_norm_b = torch.nn.utils.clip_grad_norm_(self.params_b, self.args.max_grad_norm)
            self.optimizer_a.step()
            self.optimizer_b.step()

            totals["loss_a"] += epoch_loss_a / a_action_count
            totals["loss_b"] += epoch_loss_b / b_action_count
            totals["kl_a"] += epoch_kl_a / a_action_count
            totals["kl_b"] += epoch_kl_b / b_action_count
            totals["clip_a"] += epoch_clip_a / a_action_count
            totals["clip_b"] += epoch_clip_b / b_action_count
            totals["grad_norm_a"] += float(grad_norm_a)
            totals["grad_norm_b"] += float(grad_norm_b)

        return {key: value / policy_epochs for key, value in totals.items()}

    def _transition(self, trajectory: JointTrajectory) -> str:
        draft_correct = extract_final_answer(trajectory.agent_a_draft) == trajectory.ground_truth
        final_correct = trajectory.team_reward == 1.0
        if not draft_correct and final_correct:
            return "WR_correction"
        if draft_correct and final_correct:
            return "RR_preservation"
        if draft_correct and not final_correct:
            return "RW_corruption"
        return "WW_failure"

    def _write_trajectories(self, step: int, trajectories: list[JointTrajectory]) -> None:
        with self.trajectory_path.open("a", encoding="utf-8") as f:
            for index, trajectory in enumerate(trajectories):
                row = {
                    "step": step,
                    "trajectory_index": index,
                    "question": trajectory.question,
                    "agent_a_draft": trajectory.agent_a_draft,
                    "agent_b_message": trajectory.agent_b_message,
                    "agent_a_final": trajectory.agent_a_final,
                    "draft_answer": extract_final_answer(trajectory.agent_a_draft),
                    "agent_b_answer": extract_final_answer(trajectory.agent_b_message),
                    "final_answer": extract_final_answer(trajectory.agent_a_final),
                    "ground_truth": trajectory.ground_truth,
                    "transition": self._transition(trajectory),
                    "team_reward": trajectory.team_reward,
                    "credit_a": trajectory.credit_a,
                    "credit_b": trajectory.credit_b,
                    "advantage_a": trajectory.advantage_a,
                    "advantage_b": trajectory.advantage_b,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _behavior_metrics(self, trajectories: list[JointTrajectory]) -> dict[str, float]:
        transitions = [self._transition(t) for t in trajectories]
        n = len(trajectories)
        draft_correct = [
            float(extract_final_answer(t.agent_a_draft) == t.ground_truth) for t in trajectories
        ]
        b_correct = [
            float(extract_final_answer(t.agent_b_message) == t.ground_truth) for t in trajectories
        ]
        return {
            "draft_accuracy": sum(draft_correct) / n,
            "agent_b_accuracy": sum(b_correct) / n,
            "final_accuracy": sum(t.team_reward for t in trajectories) / n,
            "correction_rate": transitions.count("WR_correction") / n,
            "preservation_rate": transitions.count("RR_preservation") / n,
            "corruption_rate": transitions.count("RW_corruption") / n,
            "failure_rate": transitions.count("WW_failure") / n,
        }

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics, indent=2))
        if self.wandb is not None:
            self.wandb.log(metrics, step=int(metrics["step"]))

    def save(self, suffix: str = "final") -> None:
        path = self.output_dir / suffix
        path.mkdir(parents=True, exist_ok=True)
        self.model_a.save_pretrained(path / "agent_a")
        self.model_b.save_pretrained(path / "agent_b")
        self.tokenizer_a.save_pretrained(path / "agent_a")
        self.tokenizer_b.save_pretrained(path / "agent_b")

    def train(self) -> None:
        questions_per_update = max(1, int(self.args.per_device_train_batch_size))
        step = 0
        epochs = max(1, int(math.ceil(float(self.args.num_train_epochs))))

        for epoch in range(epochs):
            for start in range(0, len(self.dataset), questions_per_update):
                batch = self.dataset.select(
                    range(start, min(start + questions_per_update, len(self.dataset)))
                )
                trajectories = []
                for row in batch:
                    for _ in range(int(self.args.num_generations)):
                        trajectories.append(
                            self._rollout(row["question"], row["ground_truth"])
                        )

                team_rewards = torch.tensor(
                    [t.team_reward for t in trajectories], dtype=torch.float32
                )
                credits = self.credit_assigner(team_rewards)
                advantage_a, zero_std_a = group_relative_advantage(
                    credits.agent_a, int(self.args.num_generations)
                )
                advantage_b, zero_std_b = group_relative_advantage(
                    credits.agent_b, int(self.args.num_generations)
                )

                for i, trajectory in enumerate(trajectories):
                    trajectory.credit_a = float(credits.agent_a[i])
                    trajectory.credit_b = float(credits.agent_b[i])
                    trajectory.advantage_a = float(advantage_a[i])
                    trajectory.advantage_b = float(advantage_b[i])

                self._write_trajectories(step, trajectories)
                optimization = self._update_policies(trajectories)
                behavior = self._behavior_metrics(trajectories)
                metrics = {
                    "step": step,
                    "epoch": epoch,
                    "credit_mode": self.credit_assigner.name,
                    "team_reward": team_rewards.mean().item(),
                    "credit_a_mean": credits.agent_a.mean().item(),
                    "credit_b_mean": credits.agent_b.mean().item(),
                    "advantage_a_mean": advantage_a.mean().item(),
                    "advantage_b_mean": advantage_b.mean().item(),
                    "advantage_a_std": advantage_a.std(unbiased=False).item(),
                    "advantage_b_std": advantage_b.std(unbiased=False).item(),
                    "zero_std_group_fraction_a": zero_std_a,
                    "zero_std_group_fraction_b": zero_std_b,
                    **behavior,
                    **optimization,
                }
                self._log_metrics(metrics)

                step += 1
                if self.save_steps > 0 and step % self.save_steps == 0:
                    self.save(f"checkpoint-{step}")

        self.save("final")
        if self.wandb is not None:
            self.wandb.finish()
