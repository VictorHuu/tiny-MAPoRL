from __future__ import annotations

import argparse

from trl import GRPOConfig

from tiny_maporl.credit import DiscountedInfluenceCredit, SharedTeamCredit
from tiny_maporl.data import load_gsm8k_dataset
from tiny_maporl.joint_trainer import JointGRPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", default=None)
    parser.add_argument("--device-a", default="cuda:0")
    parser.add_argument("--device-b", default="cuda:0")
    parser.add_argument("--output-dir", default="outputs/joint-grpo")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--questions-per-update", type=int, default=1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--max-completion-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.002)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--credit", choices=["shared", "discounted"], default="shared")
    parser.add_argument("--credit-discount", type=float, default=0.3)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--report-to", choices=["none", "wandb"], default="none")
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    dataset = load_gsm8k_dataset("train", max_samples=cli.max_samples)
    model_b = cli.model_b or cli.model_a
    credit_assigner = (
        SharedTeamCredit()
        if cli.credit == "shared"
        else DiscountedInfluenceCredit(cli.credit_discount)
    )

    config = GRPOConfig(
        output_dir=cli.output_dir,
        learning_rate=cli.learning_rate,
        per_device_train_batch_size=cli.questions_per_update,
        generation_batch_size=cli.num_generations,
        num_train_epochs=cli.num_train_epochs,
        num_generations=cli.num_generations,
        num_iterations=cli.num_iterations,
        max_completion_length=cli.max_completion_length,
        beta=cli.beta,
        epsilon=cli.epsilon,
        temperature=cli.temperature,
        top_p=cli.top_p,
        top_k=0,
        report_to="none",
    )

    trainer = JointGRPOTrainer(
        model_a_path=cli.model_a,
        model_b_path=model_b,
        train_dataset=dataset,
        args=config,
        device_a=cli.device_a,
        device_b=cli.device_b,
        lora_r=cli.lora_r,
        lora_alpha=cli.lora_alpha,
        load_in_4bit=not cli.no_4bit,
        credit_assigner=credit_assigner,
        report_to=cli.report_to,
        save_steps=cli.save_steps,
    )
    trainer.train()


if __name__ == "__main__":
    main()
