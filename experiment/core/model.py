
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from experiment.core.utils import softmax


@dataclass
class StepCache:
    prompt: np.ndarray
    prev_token: int
    position: int
    hidden_pre: np.ndarray
    hidden: np.ndarray
    probs: np.ndarray
    token: int


class ContextualPolicy:
    def __init__(self, prompt_dim: int, vocab_size: int, hidden_size: int, sequence_length: int, seed: int = 0):
        self.prompt_dim = prompt_dim
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        rng = np.random.default_rng(seed)
        scale = 0.10
        self.params = {
            'W_prompt': rng.normal(0.0, scale, size=(hidden_size, prompt_dim)),
            'W_prev': rng.normal(0.0, scale, size=(hidden_size, vocab_size)),
            'W_pos': rng.normal(0.0, scale, size=(hidden_size, sequence_length)),
            'b_h': np.zeros(hidden_size),
            'W_out': rng.normal(0.0, scale, size=(vocab_size, hidden_size)),
            'b_out': np.zeros(vocab_size),
        }

    def clone(self) -> 'ContextualPolicy':
        other = ContextualPolicy(self.prompt_dim, self.vocab_size, self.hidden_size, self.sequence_length)
        other.params = {k: v.copy() for k, v in self.params.items()}
        return other

    def init_grads(self) -> dict[str, np.ndarray]:
        return {k: np.zeros_like(v) for k, v in self.params.items()}

    def step_distribution(self, prompt: np.ndarray, prev_token: int, position: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hidden_pre = (
            self.params['W_prompt'] @ prompt
            + self.params['W_prev'][:, prev_token]
            + self.params['W_pos'][:, position]
            + self.params['b_h']
        )
        hidden = np.tanh(hidden_pre)
        logits = self.params['W_out'] @ hidden + self.params['b_out']
        probs = softmax(logits)
        return hidden_pre, hidden, probs

    def rollout(self, prompt: np.ndarray, bos_token: int, temperature: float = 1.0, sample: bool = True, allowed_token_ids: list[list[int]] | None = None) -> dict[str, Any]:
        tokens = []
        logprobs = []
        entropies = []
        caches: list[StepCache] = []
        prev = bos_token
        for position in range(self.sequence_length):
            hidden_pre, hidden, probs = self.step_distribution(prompt, prev, position)
            if allowed_token_ids is not None:
                mask = np.zeros_like(probs)
                mask[np.array(allowed_token_ids[position], dtype=int)] = 1.0
                probs = probs * mask
                probs = probs / (probs.sum() + 1e-12)
            if temperature != 1.0:
                adjusted = np.log(probs + 1e-12) / temperature
                probs = softmax(adjusted)
            entropy = float(-(probs * np.log(probs + 1e-12)).sum())
            token = int(np.random.choice(len(probs), p=probs)) if sample else int(np.argmax(probs))
            logprob = float(np.log(probs[token] + 1e-12))
            tokens.append(token)
            logprobs.append(logprob)
            entropies.append(entropy)
            caches.append(StepCache(prompt=prompt, prev_token=prev, position=position, hidden_pre=hidden_pre, hidden=hidden, probs=probs, token=token))
            prev = token
        return {
            'tokens': tokens,
            'logprobs': logprobs,
            'entropies': entropies,
            'caches': caches,
        }

    def accumulate_pg_gradient(self, grads: dict[str, np.ndarray], caches: list[StepCache], advantages: list[float]) -> list[float]:
        score_norms = []
        for cache, coeff in zip(caches, advantages):
            if abs(coeff) < 1e-12:
                score_norms.append(0.0)
                continue
            probs = cache.probs
            target = np.zeros_like(probs)
            target[cache.token] = 1.0
            dlogits = -coeff * (target - probs)
            score_norms.append(float(np.linalg.norm(dlogits)))
            grads['W_out'] += np.outer(dlogits, cache.hidden)
            grads['b_out'] += dlogits
            dhidden = self.params['W_out'].T @ dlogits
            dhidden_pre = dhidden * (1.0 - cache.hidden ** 2)
            grads['W_prompt'] += np.outer(dhidden_pre, cache.prompt)
            grads['W_prev'][:, cache.prev_token] += dhidden_pre
            grads['W_pos'][:, cache.position] += dhidden_pre
            grads['b_h'] += dhidden_pre
        return score_norms

    def apply_gradients(self, grads: dict[str, np.ndarray], learning_rate: float, grad_clip: float | None = None, batch_scale: float = 1.0) -> float:
        total_norm_sq = 0.0
        for grad in grads.values():
            total_norm_sq += float(np.sum((grad / batch_scale) ** 2))
        total_norm = total_norm_sq ** 0.5
        scale = 1.0
        if grad_clip and total_norm > grad_clip:
            scale = grad_clip / (total_norm + 1e-12)
        for name in self.params:
            self.params[name] -= learning_rate * (grads[name] / batch_scale) * scale
        return total_norm * scale
