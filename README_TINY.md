# tiny-MAPoRL: modern TRL path

This directory-level path is a clean-room simplification of the original MAPoRL research code. The legacy implementation remains untouched and serves only as an implementation reference.

## Scope of this first milestone

- Upstream `trl==1.12.0` instead of the vendored historical TRL fork.
- Stable `GRPOTrainer` agent-training API instead of extending the experimental PPO trainer.
- Two-agent interaction through TRL `environment_factory`.
- One **active** trainable agent and one **frozen peer** per phase.
- The active agent drafts a solution, calls `ask_partner`, receives LLM feedback, and then produces a final answer.
- GSM8K exact-match reward; no learned verifier and no reward server.
- QLoRA for the active policy and 4-bit loading for the peer model.

This is intentionally an **alternating post-co-training baseline**, not yet a claim of exact MAPoRL reproduction. After one phase trains Agent A with Agent B frozen, the saved A adapter can be loaded as the frozen peer while training B in the next phase. A later milestone will automate this alternation and add explicit agent/turn credit assignment.

## Install

Use Python 3.11 in a fresh environment:

```bash
pip install -r requirements-tiny.txt
```

TRL 1.12's `environment_factory` requires Transformers >= 5.2; the pinned TRL package resolves a compatible stack.

## Dataset-only smoke test

```bash
python train_tiny_grpo.py --dry-run --max-samples 8
```

This downloads GSM8K, constructs conversational prompts, and checks the ground-truth extraction path without loading either LLM.

## One-GPU model smoke test

Start small. The default uses Qwen3-0.6B for both the active policy and the frozen peer. Both are loaded in 4-bit, while only the active policy receives LoRA updates.

```bash
CUDA_VISIBLE_DEVICES=0 python train_tiny_grpo.py \
  --max-samples 64 \
  --num-generations 2 \
  --gradient-accumulation-steps 4 \
  --max-completion-length 192 \
  --partner-max-new-tokens 96 \
  --output-dir outputs/smoke-a
```

Keep this single-process initially. TRL creates one environment object per rollout, but those environments share one frozen partner model instance.

## Low-compute research configuration

After the smoke test, move to Qwen3-1.7B or another small instruct-capable model and increase the rollout group size. Example:

```bash
CUDA_VISIBLE_DEVICES=0 python train_tiny_grpo.py \
  --model Qwen/Qwen3-1.7B \
  --partner-model Qwen/Qwen3-1.7B \
  --max-samples 1024 \
  --num-generations 4 \
  --gradient-accumulation-steps 8 \
  --output-dir outputs/agent-a-phase1 \
  --report-to wandb
```

The first scaling target is **one 3090 per run**, not multi-GPU model parallelism. With up to four 3090s available, use the other cards for independent seeds/evaluations first. A dedicated inference worker for the peer is a later optimization if peer generation becomes the bottleneck.

## Current interaction

```text
GSM8K problem
    |
    v
Active agent drafts
    |
    | ask_partner(draft)
    v
Frozen peer critiques
    |
    v
Active agent revises
    |
    v
Final boxed answer
    |
    +--> exact task reward
    +--> tiny consultation bonus
```

The consultation bonus is only shaping (`0.05` by default); exact task correctness remains the dominant reward.

## Why GRPO first?

As of TRL 1.12, `GRPOTrainer` is a stable online-RL API with first-class tool/environment training. `PPOTrainer` lives under `trl.experimental.ppo`, so the modern path uses GRPO for the main research harness while retaining the legacy MAPoRL PPO code as an algorithmic reference.

## Next milestones

1. Run the dataset smoke test and a 64-example GPU smoke run.
2. Add an alternating A -> B -> A driver that reloads LoRA adapters between phases.
3. Record peer messages and final completions as explicit trajectories.
4. Introduce `CreditAssigner` with baselines such as shared team reward, marginal agent reward, and later turn-level credit.
5. Only after correctness is established, add a separate peer inference worker or vLLM server for throughput.
