
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CodePrompt:
    operation: str
    value: int
    prompt_vector: np.ndarray
    metadata: dict[str, Any]


class ProgramTraceTask:
    name = 'program_trace'

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.style_tokens = ['STYLE_A', 'STYLE_B']
        self.plan_tokens = ['PLAN_X', 'PLAN_Y']
        self.op_tokens = ['OP_REV', 'OP_SHIFT', 'OP_PARITY', 'OP_DUP']
        self.arg_tokens = ['ARG_0', 'ARG_1']
        self.out_tokens = ['OUT_0', 'OUT_1']
        self.bos_token = '<BOS>'
        self.vocab = [self.bos_token] + self.style_tokens + self.plan_tokens + self.op_tokens + self.arg_tokens + self.out_tokens
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.operations = self.op_tokens
        self.sequence_length = 5
        self.prompt_dim = 9  # one-hot op + one-hot value bucket + bias
        self.important_positions = [0, 0, 1, 1, 1]
        self.allowed_token_ids = [
            [self.token_to_id[t] for t in self.style_tokens],
            [self.token_to_id[t] for t in self.plan_tokens],
            [self.token_to_id[t] for t in self.op_tokens],
            [self.token_to_id[t] for t in self.arg_tokens],
            [self.token_to_id[t] for t in self.out_tokens],
        ]

    def sample_prompt(self) -> CodePrompt:
        op_index = int(self.rng.integers(0, len(self.operations)))
        value = int(self.rng.integers(0, 4))
        vec = np.zeros(self.prompt_dim, dtype=float)
        vec[op_index] = 1.0
        vec[4 + value] = 1.0
        vec[-1] = 1.0
        return CodePrompt(operation=self.operations[op_index], value=value, prompt_vector=vec, metadata={'operation': self.operations[op_index], 'value': value})

    def _target(self, prompt: CodePrompt) -> tuple[str, str, str]:
        op = prompt.operation
        value = prompt.value
        if op == 'OP_REV':
            arg = f'ARG_{value % 2}'
            out = f'OUT_{(value + 1) % 2}'
        elif op == 'OP_SHIFT':
            arg = f'ARG_{(value + 1) % 2}'
            out = f'OUT_{(value // 2) % 2}'
        elif op == 'OP_PARITY':
            arg = f'ARG_{value % 2}'
            out = f'OUT_{value % 2}'
        else:  # OP_DUP
            arg = f'ARG_{(value // 2) % 2}'
            out = f'OUT_{(value + value // 2) % 2}'
        return op, arg, out

    def reward(self, prompt: CodePrompt, tokens: list[int]) -> tuple[float, dict[str, Any]]:
        decoded = [self.vocab[t] for t in tokens]
        op, arg, out = self._target(prompt)
        reward = 1.0 if decoded[2] == op and decoded[3] == arg and decoded[4] == out else 0.0
        return reward, {
            'decoded_tokens': decoded,
            'target_tokens': [decoded[0], decoded[1], op, arg, out],
            'important_positions': self.important_positions,
        }
