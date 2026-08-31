from __future__ import annotations

from datasets import Dataset, load_dataset

from .rewards import extract_gsm8k_ground_truth

_SYSTEM_PROMPT = """You are the active solver in a two-agent collaboration.
Solve the math problem carefully. Before giving your final answer, call the
ask_partner tool exactly once with your current draft solution. Use the peer's
feedback as evidence, not as authority. End with the final numeric answer in
\\boxed{...} format."""


def load_gsm8k_dataset(split: str = "train", max_samples: int | None = None) -> Dataset:
    """Load GSM8K and convert it to the conversational format expected by TRL."""
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    if max_samples is not None:
        max_samples = min(max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    def convert(row):
        question = row["question"]
        return {
            "prompt": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            # Extra columns are intentionally retained: TRL forwards them to
            # reward functions and environment.reset().
            "question": question,
            "ground_truth": extract_gsm8k_ground_truth(row["answer"]),
        }

    return dataset.map(convert, remove_columns=dataset.column_names)
