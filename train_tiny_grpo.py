from __future__ import annotations

import argparse
from functools import partial

import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from tiny_maporl.data import load_gsm8k_dataset
from tiny_maporl.environment import PeerReviewEnv
from tiny_maporl.partner import FrozenPartner
from tiny_maporl.rewards import consultation_reward, exact_answer_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one active LLM agent with a frozen peer using the stable TRL "
            "GRPO agent/environment API. Alternate active/peer checkpoints across "
            "phases for low-compute post-co-training."
        )
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--partner-model", default=None)
    parser.add_argument("--partner-adapter", default=None)
    parser.add_argument("--partner-device", default="cuda:0")
    parser.add_argument("--output-dir", default="outputs/tiny-maporl-qwen3-0.6b")
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=384)
    parser.add_argument("--partner-max-new-tokens", type=int, default=192)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--consult-bonus", type=float, default=0.05)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--report-to", choices=["none", "wandb", "tensorboard"], default="none")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_gsm8k_dataset("train", max_samples=args.max_samples)

    if args.dry_run:
        sample = dataset[0]
        print("prompt:", sample["prompt"])
        print("ground_truth:", sample["ground_truth"])
        print("samples:", len(dataset))
        return

    if not torch.cuda.is_available():
        raise RuntimeError("The training path currently expects a CUDA GPU.")

    partner_model = args.partner_model or args.model
    partner = FrozenPartner(
        model_name=partner_model,
        adapter_path=args.partner_adapter,
        device=args.partner_device,
        load_in_4bit=True,
        max_new_tokens=args.partner_max_new_tokens,
    )

    # One lightweight environment instance is created per rollout, while all of
    # them share the same frozen partner model object.
    environment_factory = partial(PeerReviewEnv, partner=partner)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    use_bf16 = bool(torch.cuda.is_bf16_supported())
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_tool_calling_iterations=2,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to=args.report_to,
        chat_template_kwargs={"enable_thinking": False},
        reward_weights=[1.0, args.consult_bonus],
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[exact_answer_reward, consultation_reward],
        environment_factory=environment_factory,
        peft_config=peft_config,
        quantization_config=quantization_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
