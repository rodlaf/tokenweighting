
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiment.core.model import ContextualPolicy
from experiment.core.utils import ensure_dir, mean_std, save_json, set_seed
from experiment.core.weighting import compute_token_weights
from experiment.tasks.arithmetic import ArithmeticTraceTask
from experiment.tasks.code_trace import ProgramTraceTask


TASKS = {
    'arithmetic_trace': ArithmeticTraceTask,
    'program_trace': ProgramTraceTask,
}


@dataclass
class ExperimentResult:
    config: dict[str, Any]
    seed: int
    output_dir: str
    train_summary: dict[str, Any]
    eval_summary: dict[str, Any]


class ExperimentRunner:
    def __init__(self, config: dict[str, Any], output_dir: str):
        self.config = config
        self.output_dir = ensure_dir(output_dir)

    def build_task(self, seed: int):
        task_name = self.config['task']['name']
        return TASKS[task_name](seed=seed)

    def build_policy(self, task, seed: int) -> ContextualPolicy:
        return ContextualPolicy(
            prompt_dim=task.prompt_dim,
            vocab_size=len(task.vocab),
            hidden_size=self.config['model']['hidden_size'],
            sequence_length=task.sequence_length,
            seed=seed,
        )

    def compute_advantages(self, rewards: list[float]) -> list[float]:
        baseline = self.config['training']['baseline']
        eps = 1e-6
        if baseline == 'rloo':
            if len(rewards) == 1:
                return rewards[:]
            total = sum(rewards)
            return [float(r - ((total - r) / max(1, len(rewards) - 1))) for r in rewards]
        if baseline == 'grpo':
            mean = float(np.mean(rewards))
            std = float(np.std(rewards))
            return [float((r - mean) / (std + eps)) for r in rewards]
        raise ValueError(f'Unknown baseline: {baseline}')

    def _sample_group(self, policy, task, prompt):
        group = []
        for _ in range(self.config['training']['group_size']):
            rollout = policy.rollout(
                prompt.prompt_vector,
                bos_token=task.token_to_id[task.bos_token],
                temperature=self.config['training'].get('temperature', 1.0),
                sample=True,
                allowed_token_ids=task.allowed_token_ids,
            )
            reward, meta = task.reward(prompt, rollout['tokens'])
            group.append({**rollout, 'reward': reward, 'meta': meta})
        return group

    def train_seed(self, seed: int) -> ExperimentResult:
        set_seed(seed)
        task = self.build_task(seed)
        policy = self.build_policy(task, seed)
        steps = self.config['training']['steps']
        prompts_per_step = self.config['training']['prompts_per_step']
        weight_mode = self.config['training']['weighting']
        learning_rate = self.config['training']['learning_rate']
        grad_clip = self.config['training'].get('grad_clip', None)
        log_interval = self.config['training'].get('log_interval', 50)
        train_history = []

        for step in range(1, steps + 1):
            grads = policy.init_grads()
            batch_rewards = []
            important_mass = []
            weighted_token_norm_vars = []
            active_advantages = []
            for _ in range(prompts_per_step):
                prompt = task.sample_prompt()
                group = self._sample_group(policy, task, prompt)
                rewards = [sample['reward'] for sample in group]
                advantages = self.compute_advantages(rewards)
                for sample, advantage in zip(group, advantages):
                    weights = compute_token_weights(weight_mode, sample['logprobs'], sample['entropies'])
                    coeffs = [advantage * w for w in weights]
                    token_score_norms = policy.accumulate_pg_gradient(grads, sample['caches'], coeffs)
                    batch_rewards.append(sample['reward'])
                    active_advantages.append(advantage)
                    mask = np.array(sample['meta']['important_positions'], dtype=float)
                    important_mass.append(float(np.dot(mask, np.array(weights))))
                    weighted_norms = np.array(token_score_norms) * np.array(weights)
                    weighted_token_norm_vars.append(float(np.var(weighted_norms)))
            grad_norm = policy.apply_gradients(grads, learning_rate=learning_rate, grad_clip=grad_clip, batch_scale=max(1, prompts_per_step * self.config['training']['group_size']))
            summary = {
                'step': step,
                'reward_mean': float(np.mean(batch_rewards)),
                'reward_std': float(np.std(batch_rewards)),
                'advantage_abs_mean': float(np.mean(np.abs(active_advantages))) if active_advantages else 0.0,
                'important_mass_mean': float(np.mean(important_mass)),
                'gradient_norm_variance': float(np.mean(weighted_token_norm_vars)),
                'grad_norm': float(grad_norm),
            }
            train_history.append(summary)
            if step % log_interval == 0 or step == 1 or step == steps:
                print(f"[seed={seed}] step {step:04d} reward={summary['reward_mean']:.3f} important_mass={summary['important_mass_mean']:.3f} grad_var={summary['gradient_norm_variance']:.5f}")

        eval_summary = self.evaluate(policy, task)
        train_summary = {
            'final_reward_mean': float(train_history[-1]['reward_mean']),
            'final_reward_std': float(train_history[-1]['reward_std']),
            'final_important_mass_mean': float(train_history[-1]['important_mass_mean']),
            'final_gradient_norm_variance': float(train_history[-1]['gradient_norm_variance']),
            'history': train_history,
        }
        seed_dir = ensure_dir(Path(self.output_dir) / f'seed_{seed}')
        save_json(seed_dir / 'train_summary.json', train_summary)
        save_json(seed_dir / 'eval_summary.json', eval_summary)
        return ExperimentResult(config=self.config, seed=seed, output_dir=str(seed_dir), train_summary=train_summary, eval_summary=eval_summary)

    def evaluate(self, policy, task) -> dict[str, Any]:
        eval_prompts = self.config['evaluation']['num_prompts']
        sample_count = self.config['evaluation'].get('passk_samples', 4)
        greedy_rewards = []
        passk_rewards = []
        important_mass = []
        examples = []
        for _ in range(eval_prompts):
            prompt = task.sample_prompt()
            greedy = policy.rollout(prompt.prompt_vector, bos_token=task.token_to_id[task.bos_token], sample=False, allowed_token_ids=task.allowed_token_ids)
            reward, meta = task.reward(prompt, greedy['tokens'])
            greedy_rewards.append(reward)
            weights = compute_token_weights(self.config['training']['weighting'], greedy['logprobs'], greedy['entropies'])
            important_mass.append(float(np.dot(np.array(meta['important_positions']), np.array(weights))))
            sampled_rewards = []
            for _ in range(sample_count):
                sample = policy.rollout(prompt.prompt_vector, bos_token=task.token_to_id[task.bos_token], temperature=self.config['training'].get('temperature', 1.0), sample=True, allowed_token_ids=task.allowed_token_ids)
                sample_reward, _ = task.reward(prompt, sample['tokens'])
                sampled_rewards.append(sample_reward)
            passk_rewards.append(1.0 if max(sampled_rewards) > 0 else 0.0)
            if len(examples) < 5:
                examples.append({'prompt': prompt.metadata, 'greedy_tokens': meta['decoded_tokens'], 'target_tokens': meta['target_tokens'], 'reward': reward})
        return {
            'greedy_accuracy': float(np.mean(greedy_rewards)),
            'pass_at_k': float(np.mean(passk_rewards)),
            'important_mass_mean': float(np.mean(important_mass)),
            'examples': examples,
        }

    def run(self) -> dict[str, Any]:
        seeds = self.config['seeds']
        results = [self.train_seed(seed) for seed in seeds]
        aggregate = {
            'task': self.config['task']['name'],
            'baseline': self.config['training']['baseline'],
            'weighting': self.config['training']['weighting'],
            'seeds': seeds,
            'greedy_accuracy': mean_std([r.eval_summary['greedy_accuracy'] for r in results]),
            'pass_at_k': mean_std([r.eval_summary['pass_at_k'] for r in results]),
            'important_mass_mean': mean_std([r.eval_summary['important_mass_mean'] for r in results]),
            'final_reward_mean': mean_std([r.train_summary['final_reward_mean'] for r in results]),
            'final_gradient_norm_variance': mean_std([r.train_summary['final_gradient_norm_variance'] for r in results]),
        }
        save_json(Path(self.output_dir) / 'aggregate_summary.json', aggregate)
        return aggregate
