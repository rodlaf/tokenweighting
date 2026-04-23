from __future__ import annotations

import argparse
import json
import math
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup

from data import TASK_SPECS, load_task_dataset
from rewards import pass_at_k, score_completions
from token_weights import (
    TokenWeightingConfig,
    build_token_weights,
    entropy_from_logits,
    weight_concentration_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ITW experiments on Qwen + GSM8K/MBPP.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config, model, and dataset without training.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str) -> dict[str, Any]:
    cfg_path = Path(path)
    config = yaml.safe_load(cfg_path.read_text())
    base = config.pop("base_config", None)
    if base:
        base_path = (cfg_path.parent / base).resolve()
        parent = load_config(str(base_path))
        config = deep_update(parent, config)
    return config


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def pick_dtype(name: str | None) -> torch.dtype:
    mapping = {
        None: torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[name]


def _load_base_model(model_cfg: dict[str, Any]):
    """Load the base pretrained model (no LoRA, no freezing)."""
    model_name = model_cfg["name_or_path"]
    dtype = pick_dtype(model_cfg.get("torch_dtype", "bfloat16"))
    load_in_4bit = model_cfg.get("load_in_4bit", True)
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    load_kwargs: dict[str, Any] = {
        "device_map": {"": 0},
        "trust_remote_code": model_cfg.get("trust_remote_code", False),
    }
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    else:
        load_kwargs["dtype"] = dtype
    return AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)


def build_model_and_tokenizer(model_cfg: dict[str, Any]):
    model_name = model_cfg["name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_base_model(model_cfg)

    uses_lora = False
    lora_cfg = model_cfg.get("lora", {})
    if lora_cfg.get("enabled", True):
        peft_config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        uses_lora = True

    return model, tokenizer, uses_lora


def build_ref_model(model_cfg: dict[str, Any]):
    """Load a frozen copy of the base model for divergence weighting (non-LoRA path)."""
    model = _load_base_model(model_cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def sample_batch(dataset, batch_size: int, rng: random.Random) -> list[dict[str, Any]]:
    indices = [rng.randrange(len(dataset)) for _ in range(batch_size)]
    return [dataset[int(idx)] for idx in indices]


def build_completion_mask(tokens: torch.Tensor, pad_token_id: int | None, eos_token_id: int | None) -> torch.Tensor:
    mask = torch.ones_like(tokens, dtype=torch.float32)
    if pad_token_id is not None:
        mask = mask * tokens.ne(pad_token_id)
    if eos_token_id is None:
        return mask.float()
    for row in range(tokens.size(0)):
        eos_positions = (tokens[row] == eos_token_id).nonzero(as_tuple=False)
        if eos_positions.numel() > 0:
            first = int(eos_positions[0].item())
            if first + 1 < tokens.size(1):
                mask[row, first + 1 :] = 0
    return mask.float()


def compute_advantages(rewards: torch.Tensor, algorithm: str, eps: float = 1e-8) -> torch.Tensor:
    if algorithm == "rloo":
        if rewards.size(1) < 2:
            raise ValueError("RLOO needs num_generations >= 2")
        baseline = (rewards.sum(dim=1, keepdim=True) - rewards) / (rewards.size(1) - 1)
        return rewards - baseline
    if algorithm == "grpo":
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True).clamp_min(eps)
        return (rewards - mean) / std
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def gather_completion_logps(model, sequences: torch.Tensor, prompt_width: int, pad_token_id: int | None, return_hidden: bool = False):
    attention_mask = torch.ones_like(sequences, dtype=torch.long)
    if pad_token_id is not None:
        attention_mask = sequences.ne(pad_token_id).long()
    outputs = model(input_ids=sequences[:, :-1], attention_mask=attention_mask[:, :-1], use_cache=False, output_hidden_states=return_hidden)
    logits = outputs.logits[:, prompt_width - 1 :, :]
    labels = sequences[:, prompt_width:]
    logits_f = logits.float()
    token_logits = torch.gather(logits_f, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    log_Z = torch.logsumexp(logits_f, dim=-1)
    token_logps = token_logits - log_Z
    del logits_f, token_logits, log_Z
    entropies = entropy_from_logits(logits.detach())
    if return_hidden:
        hidden = outputs.hidden_states[-1][:, prompt_width - 1 :, :]
        return token_logps, entropies, hidden
    return token_logps, entropies


def gather_ref_logps(ref_model, sequences: torch.Tensor, prompt_width: int, pad_token_id: int | None) -> torch.Tensor:
    """Forward pass on a reference model to get base log-probs."""
    attention_mask = torch.ones_like(sequences, dtype=torch.long)
    if pad_token_id is not None:
        attention_mask = sequences.ne(pad_token_id).long()
    with torch.no_grad():
        outputs = ref_model(input_ids=sequences[:, :-1], attention_mask=attention_mask[:, :-1], use_cache=False)
    logits = outputs.logits[:, prompt_width - 1 :, :]
    labels = sequences[:, prompt_width:]
    logits_f = logits.float()
    token_logits = torch.gather(logits_f, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    log_Z = torch.logsumexp(logits_f, dim=-1)
    return (token_logits - log_Z).detach()


def gather_base_hidden_states(model, sequences: torch.Tensor, prompt_width: int, pad_token_id: int | None, uses_lora: bool) -> torch.Tensor:
    """Run a forward pass with the adapter disabled (or plain model) to obtain base-model hidden states."""
    attention_mask = torch.ones_like(sequences, dtype=torch.long)
    if pad_token_id is not None:
        attention_mask = sequences.ne(pad_token_id).long()
    with torch.no_grad():
        if uses_lora:
            with model.disable_adapter():
                outputs = model(
                    input_ids=sequences[:, :-1],
                    attention_mask=attention_mask[:, :-1],
                    use_cache=False,
                    output_hidden_states=True,
                )
        else:
            outputs = model(
                input_ids=sequences[:, :-1],
                attention_mask=attention_mask[:, :-1],
                use_cache=False,
                output_hidden_states=True,
            )
    return outputs.hidden_states[-1][:, prompt_width - 1 :, :].detach()


def adapter_residual_norms_from_hidden(adapted_hidden: torch.Tensor, base_hidden: torch.Tensor) -> torch.Tensor:
    """Compute ||h_adapted_t - h_base_t||_2 at each token position."""
    delta = adapted_hidden.float() - base_hidden.float()
    return delta.norm(dim=-1)


def generate_group(
    model,
    tokenizer,
    prompts: list[str],
    *,
    num_generations: int,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    repeated = {k: v.repeat_interleave(num_generations, dim=0) for k, v in encoded.items()}
    prompt_width = repeated["input_ids"].size(1)

    was_training = model.training
    model.eval()
    generate_kwargs = dict(
        **repeated,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
    with torch.no_grad():
        outputs = model.generate(**generate_kwargs)
    if was_training:
        model.train()

    sequences = outputs.sequences
    completion_ids = sequences[:, prompt_width:]
    completion_mask = build_completion_mask(completion_ids, tokenizer.pad_token_id, tokenizer.eos_token_id).to(model.device)
    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    return repeated, sequences, completion_ids, completion_mask, completions, prompt_width


def compute_loss(
    model,
    tokenizer,
    sequences: torch.Tensor,
    prompt_width: int,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    weighting_mode: str,
    sharpening: float = 1.0,
    ref_model=None,
    uses_lora: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    need_hidden = weighting_mode == "adapter_residual"
    result = gather_completion_logps(model, sequences, prompt_width, tokenizer.pad_token_id, return_hidden=need_hidden)
    if need_hidden:
        token_logps, entropies, hidden_states = result
    else:
        token_logps, entropies = result

    steps = min(token_logps.size(1), completion_mask.size(1))
    token_logps = token_logps[:, :steps]
    entropies = entropies[:, :steps]
    completion_mask = completion_mask[:, :steps]

    ref_token_logps = None
    if weighting_mode == "divergence":
        if ref_model is not None:
            ref_token_logps = gather_ref_logps(ref_model, sequences, prompt_width, tokenizer.pad_token_id)
        elif uses_lora:
            with model.disable_adapter():
                ref_token_logps = gather_ref_logps(model, sequences, prompt_width, tokenizer.pad_token_id)
        else:
            raise ValueError("divergence weighting requires ref_model when LoRA is disabled")
        ref_token_logps = ref_token_logps[:, :steps]

    adapter_residual_norms = None
    if weighting_mode == "adapter_residual":
        if not uses_lora:
            raise ValueError("adapter_residual weighting requires LoRA to be enabled")
        adapted_hidden = hidden_states[:, :steps, :].detach()
        base_hidden = gather_base_hidden_states(model, sequences, prompt_width, tokenizer.pad_token_id, uses_lora=True)[:, :steps, :]
        adapter_residual_norms = adapter_residual_norms_from_hidden(adapted_hidden, base_hidden).detach()

    weights = build_token_weights(
        TokenWeightingConfig(mode=weighting_mode, sharpening=sharpening),
        completion_mask,
        per_token_logps=token_logps.detach(),
        entropies=entropies,
        ref_token_logps=ref_token_logps,
        adapter_residual_norms=adapter_residual_norms,
    )
    objective = (weights * token_logps * completion_mask).sum(dim=-1)
    loss = -(advantages.to(objective.dtype) * objective).mean()
    metrics = {
        "mean_weight_on_nonzero": float(weights[completion_mask > 0].mean().item()) if (completion_mask > 0).any() else 0.0,
        "mean_completion_length": float(completion_mask.sum(dim=-1).float().mean().item()),
        "mean_surprisal": float((-token_logps.detach() * completion_mask).sum().item() / completion_mask.sum().clamp_min(1).item()),
        **weight_concentration_metrics(weights.detach(), completion_mask),
    }
    if adapter_residual_norms is not None:
        nonzero = completion_mask > 0
        if nonzero.any():
            metrics["mean_adapter_residual_norm"] = float(adapter_residual_norms[nonzero].mean().item())
            metrics["max_adapter_residual_norm"] = float(adapter_residual_norms[nonzero].max().item())
    return loss, metrics


def evaluate(model, tokenizer, dataset, config: dict[str, Any]) -> dict[str, Any]:
    eval_cfg = config["evaluation"]
    train_cfg = config["training"]
    examples = [dataset[i] for i in range(min(eval_cfg["num_examples"], len(dataset)))]
    prompts = [example["prompt"] for example in examples]

    _, _, _, _, greedy_texts, _ = generate_group(
        model,
        tokenizer,
        prompts,
        num_generations=1,
        max_prompt_length=train_cfg["max_prompt_length"],
        max_new_tokens=train_cfg["max_new_tokens"],
        temperature=train_cfg.get("temperature", 0.8),
        top_p=train_cfg.get("top_p", 0.95),
        do_sample=False,
    )
    greedy_rewards = score_completions(examples, greedy_texts, timeout=eval_cfg.get("code_timeout", 3.0))

    _, _, _, _, sampled_texts, _ = generate_group(
        model,
        tokenizer,
        prompts,
        num_generations=eval_cfg["pass_k"],
        max_prompt_length=train_cfg["max_prompt_length"],
        max_new_tokens=train_cfg["max_new_tokens"],
        temperature=train_cfg.get("temperature", 0.8),
        top_p=train_cfg.get("top_p", 0.95),
        do_sample=True,
    )
    repeated_examples = [example for example in examples for _ in range(eval_cfg["pass_k"])]
    sampled_rewards = score_completions(repeated_examples, sampled_texts, timeout=eval_cfg.get("code_timeout", 3.0))
    grouped_rewards = [sampled_rewards[i : i + eval_cfg["pass_k"]] for i in range(0, len(sampled_rewards), eval_cfg["pass_k"])]

    return {
        "greedy_accuracy": sum(greedy_rewards) / len(greedy_rewards),
        f"pass@{eval_cfg['pass_k']}": pass_at_k(grouped_rewards, eval_cfg["pass_k"]),
        "num_eval_examples": len(examples),
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def maybe_save_checkpoint(model, tokenizer, output_dir: Path, step: int) -> None:
    ckpt_dir = ensure_dir(output_dir / f"checkpoint-{step}")
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)


def run_training(config: dict[str, Any], dry_run: bool) -> None:
    seed_everything(config["seed"])
    output_dir = ensure_dir(Path(config["output_dir"]))
    save_json(output_dir / "resolved_config.json", config)

    model, tokenizer, uses_lora = build_model_and_tokenizer(config["model"])

    ref_model = None
    weighting = config["training"]["weighting"]
    if weighting == "divergence" and not uses_lora:
        print("Loading frozen reference model for divergence weighting...")
        ref_model = build_ref_model(config["model"])

    train_ds = load_task_dataset(config["dataset"]["name"], config["dataset"]["train_split"], max_samples=config["dataset"].get("max_train_samples"))
    eval_ds = load_task_dataset(config["dataset"]["name"], config["dataset"]["eval_split"], max_samples=config["dataset"].get("max_eval_samples"))

    summary = {
        "model": config["model"]["name_or_path"],
        "dataset": config["dataset"]["name"],
        "algorithm": config["training"]["algorithm"],
        "weighting": weighting,
        "uses_lora": uses_lora,
        "sharpening": config["training"].get("sharpening", 1.0),
        "train_examples": len(train_ds),
        "eval_examples": len(eval_ds),
        "dry_run": dry_run,
    }
    print(json.dumps(summary, indent=2))
    if dry_run:
        return

    train_cfg = config["training"]

    model.train()
    optimizer = AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg.get("weight_decay", 0.0))
    total_steps = train_cfg["train_steps"]
    warmup_steps = min(train_cfg.get("warmup_steps", 0), total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    rng = random.Random(config["seed"])
    metrics_path = output_dir / "train_metrics.jsonl"

    grad_accum = train_cfg.get("grad_accum_steps", 1)
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, total_steps + 1):
        step_rewards = []
        step_losses = []
        step_started = time.time()

        for micro in range(grad_accum):
            batch = sample_batch(train_ds, train_cfg["batch_size"], rng)
            prompts = [example["prompt"] for example in batch]
            repeated_examples = [example for example in batch for _ in range(train_cfg["num_generations"])]

            _, sequences, _, completion_mask, completions, prompt_width = generate_group(
                model,
                tokenizer,
                prompts,
                num_generations=train_cfg["num_generations"],
                max_prompt_length=train_cfg["max_prompt_length"],
                max_new_tokens=train_cfg["max_new_tokens"],
                temperature=train_cfg.get("temperature", 0.8),
                top_p=train_cfg.get("top_p", 0.95),
                do_sample=True,
            )
            rewards = score_completions(repeated_examples, completions, timeout=config["evaluation"].get("code_timeout", 3.0))
            reward_tensor = torch.tensor(rewards, device=model.device, dtype=torch.float32).view(len(batch), train_cfg["num_generations"])

            advantages = compute_advantages(reward_tensor, train_cfg["algorithm"]).reshape(-1)
            loss, aux = compute_loss(
                model,
                tokenizer,
                sequences,
                prompt_width,
                completion_mask,
                advantages,
                train_cfg["weighting"],
                sharpening=train_cfg.get("sharpening", 1.0),
                ref_model=ref_model,
                uses_lora=uses_lora,
            )
            (loss / grad_accum).backward()
            step_rewards.extend(rewards)
            step_losses.append(float(loss.item()))

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("max_grad_norm", 1.0))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        record = {
            "step": step,
            "loss": sum(step_losses) / len(step_losses),
            "reward_mean": sum(step_rewards) / len(step_rewards),
            "reward_std": float(torch.tensor(step_rewards, dtype=torch.float32).std(unbiased=False).item()),
            "learning_rate": scheduler.get_last_lr()[0],
            "grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
            "seconds": round(time.time() - step_started, 3),
            **aux,
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record))

        eval_every = train_cfg.get("eval_every")
        if eval_every and step % eval_every == 0:
            eval_metrics = evaluate(model, tokenizer, eval_ds, config)
            save_json(output_dir / f"eval_step_{step}.json", eval_metrics)
            print(json.dumps({"step": step, **eval_metrics}))

        save_every = train_cfg.get("save_every")
        if save_every and step % save_every == 0:
            maybe_save_checkpoint(model, tokenizer, output_dir, step)

    maybe_save_checkpoint(model, tokenizer, output_dir, total_steps)
    save_json(output_dir / "final_eval.json", evaluate(model, tokenizer, eval_ds, config))


def validate_config(config: dict[str, Any]) -> None:
    required_top = {"seed", "output_dir", "model", "dataset", "training", "evaluation"}
    missing = required_top - set(config)
    if missing:
        raise ValueError(f"Missing top-level config keys: {sorted(missing)}")
    if config["dataset"]["name"] not in TASK_SPECS:
        raise ValueError(f"Unsupported dataset: {config['dataset']['name']}")
    if config["training"]["algorithm"] not in {"rloo", "grpo"}:
        raise ValueError("training.algorithm must be 'rloo' or 'grpo'")
    if config["training"]["weighting"] not in {
        "uniform",
        "surprisal",
        "entropy_reduction",
        "divergence",
        "adapter_residual",
    }:
        raise ValueError(
            "training.weighting must be uniform, surprisal, entropy_reduction, divergence, or adapter_residual"
        )
    if config["training"]["num_generations"] < 2:
        raise ValueError("training.num_generations must be >= 2")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_config(config)
    run_training(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
