
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ArithmeticPrompt:
    a: int
    b: int
    prompt_vector: np.ndarray
    metadata: dict[str, Any]


class ArithmeticTraceTask:
    name = 'arithmetic_trace'

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.style_tokens = ['STYLE_A', 'STYLE_B']
        self.plan_tokens = ['PLAN_X', 'PLAN_Y']
        self.carry_tokens = ['C0', 'C1']
        self.parity_tokens = ['P0', 'P1']
        self.bucket_tokens = [f'B{i}' for i in range(4)]
        self.bos_token = '<BOS>'
        self.vocab = [self.bos_token] + self.style_tokens + self.plan_tokens + self.carry_tokens + self.parity_tokens + self.bucket_tokens
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.sequence_length = 5
        self.prompt_dim = 17  # one-hot a, one-hot b, bias
        self.important_positions = [0, 0, 1, 1, 1]
        self.allowed_token_ids = [
            [self.token_to_id[t] for t in self.style_tokens],
            [self.token_to_id[t] for t in self.plan_tokens],
            [self.token_to_id[t] for t in self.carry_tokens],
            [self.token_to_id[t] for t in self.parity_tokens],
            [self.token_to_id[t] for t in self.bucket_tokens],
        ]

    def sample_prompt(self) -> ArithmeticPrompt:
        a = int(self.rng.integers(0, 8))
        b = int(self.rng.integers(0, 8))
        vec = np.zeros(self.prompt_dim, dtype=float)
        vec[a] = 1.0
        vec[8 + b] = 1.0
        vec[-1] = 1.0
        return ArithmeticPrompt(a=a, b=b, prompt_vector=vec, metadata={'a': a, 'b': b})

    def reward(self, prompt: ArithmeticPrompt, tokens: list[int]) -> tuple[float, dict[str, Any]]:
        decoded = [self.vocab[t] for t in tokens]
        total = prompt.a + prompt.b
        carry = 1 if total >= 8 else 0
        parity = total % 2
        bucket = min(total // 4, 3)
        carry_ok = decoded[2] == f'C{carry}'
        parity_ok = decoded[3] == f'P{parity}'
        bucket_ok = decoded[4] == f'B{bucket}'
        reward = 1.0 if carry_ok and parity_ok and bucket_ok else 0.0
        return reward, {
            'decoded_tokens': decoded,
            'target_tokens': [decoded[0], decoded[1], f'C{carry}', f'P{parity}', f'B{bucket}'],
            'correct_answer': total,
            'important_positions': self.important_positions,
        }
