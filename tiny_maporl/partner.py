from __future__ import annotations

import threading

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class FrozenPartner:
    """Small frozen LLM used as the peer in one alternating co-training phase."""

    def __init__(
        self,
        model_name: str,
        adapter_path: str | None = None,
        device: str = "cuda:0",
        load_in_4bit: bool = True,
        max_new_tokens: int = 192,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._lock = threading.Lock()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"dtype": torch.bfloat16}
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = {"": device}

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if not load_in_4bit:
            self.model.to(device)
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def review(self, question: str, draft: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the peer reviewer in a two-agent math team. Check the other "
                    "agent's draft for concrete reasoning or arithmetic errors. Be concise. "
                    "If it is correct, say why. If it is wrong, identify the first useful correction."
                ),
            },
            {
                "role": "user",
                "content": f"Problem:\n{question}\n\nOther agent draft:\n{draft}",
            },
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.model.device)

        # Environment instances may be used concurrently by TRL rollouts. Keep
        # generation serialized until a dedicated inference worker is added.
        with self._lock:
            output = self.model.generate(
                inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0, inputs.shape[-1] :], skip_special_tokens=True).strip()
