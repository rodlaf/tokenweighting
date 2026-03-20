from __future__ import annotations

import contextlib
import io
import math
import multiprocessing as mp
import queue
import re
import textwrap
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_IMPORT_BLOCK = "import math\nimport collections\nimport itertools\nimport functools\nimport heapq\nimport bisect\nimport random\nimport statistics\n"


@dataclass
class ExecutionResult:
    passed: bool
    error: Optional[str] = None


def extract_python_code(prediction: str) -> str:
    if prediction is None:
        return ""
    matches = _CODE_BLOCK_RE.findall(prediction)
    code = matches[-1] if matches else prediction
    code = code.replace("\r\n", "\n")
    return textwrap.dedent(code).strip()


def _apply_resource_limits() -> None:
    if resource is None:
        return
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    except Exception:
        pass
    try:
        memory_limit = 1_500_000_000
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    except Exception:
        pass
    try:
        file_limit = 1_000_000
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
    except Exception:
        pass



def _run_candidate(code: str, tests: Sequence[str], setup_code: Optional[str], result_queue: mp.Queue) -> None:
    _apply_resource_limits()
    ns = {"__name__": "__main__"}
    payload_parts = [_IMPORT_BLOCK]
    if setup_code:
        payload_parts.append(setup_code)
    payload_parts.append(code)
    payload_parts.extend(tests)
    payload = "\n\n".join(part for part in payload_parts if part)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(payload, "<mbpp>", "exec"), ns, ns)
        result_queue.put(ExecutionResult(passed=True))
    except BaseException as exc:
        result_queue.put(ExecutionResult(passed=False, error=f"{type(exc).__name__}: {exc}"))


def run_with_timeout(code: str, tests: Sequence[str], setup_code: Optional[str] = None, timeout: float = 3.0) -> ExecutionResult:
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_run_candidate, args=(code, list(tests), setup_code, result_queue))
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return ExecutionResult(passed=False, error=f"Timeout after {timeout:.2f}s")

    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return ExecutionResult(passed=False, error="No result returned from worker")


def mbpp_pass_fail_reward(prediction: str, tests: Sequence[str], setup_code: Optional[str] = None, timeout: float = 3.0) -> int:
    code = extract_python_code(prediction)
    if not code.strip():
        return 0
    result = run_with_timeout(code=code, tests=tests, setup_code=setup_code, timeout=timeout)
    return int(result.passed)


def estimate_pass_at_k(n: int, c: int, k: int) -> Optional[float]:
    if n <= 0 or k <= 0 or k > n:
        return None
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def score_pass_at_k(sampled_rewards: Iterable[int], k_values: Sequence[int]) -> dict[str, Optional[float]]:
    rewards = list(sampled_rewards)
    n = len(rewards)
    c = sum(int(r > 0) for r in rewards)
    return {f"pass@{k}": estimate_pass_at_k(n=n, c=c, k=k) for k in k_values}
