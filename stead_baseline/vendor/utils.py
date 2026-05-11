"""Reconstructed `utils.py` for STEAD.

This file is NOT part of the upstream STEAD release. It is local glue
added to make the vendored STEAD code importable.

Background: the upstream repository
  https://github.com/7-yaya/STEAD @ dc0a48ee7ba98f8e80b047b33105120919bb40a9
ships `stead.py` and `main_stead.py` (by the STEAD authors — Yuang Qi et al.,
USTC) which both `import` from a module called `utils`. That `utils.py` is
absent from the published repo. This file reconstructs only the helpers
imported by the vendored modules:
- `top_k_logits`, `top_p_logits` — standard logit filters
- `set_seed` — torch/numpy/random determinism
- `SingleExampleOutput` — re-exported from `stego_utils.py`
- `get_probs_indices_past` — imported by `stead.py` but never called in any
  path our wrapper exercises; provided as a loud stub

Source for the logit filters is the standard HuggingFace `transformers`
implementation pattern, kept dependency-free.
"""

from __future__ import annotations

import random as _random
from typing import Optional

import numpy as _np
import torch
from torch.nn import functional as F

# Re-export so `from utils import SingleExampleOutput` resolves.
from stego_utils import SingleExampleOutput  # noqa: F401


def top_k_logits(logits: torch.Tensor, k: Optional[int]) -> torch.Tensor:
    """Keep the top-k logits per row; set the rest to -inf."""
    if k is None or k <= 0:
        return logits
    values, _ = torch.topk(logits, k, dim=-1)
    threshold = values[..., -1:].expand_as(logits)
    return torch.where(
        logits < threshold,
        torch.full_like(logits, float("-inf")),
        logits,
    )


def top_p_logits(logits: torch.Tensor, p: Optional[float]) -> torch.Tensor:
    """Nucleus filter: drop tokens outside the smallest set summing to >= p."""
    if p is None or p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_remove = cumulative > p
    # Shift right by one so we always keep at least the top token.
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False
    to_remove = sorted_remove.scatter(-1, sorted_indices, sorted_remove)
    return logits.masked_fill(to_remove, float("-inf"))


def set_seed(seed: int) -> None:
    """Determinism across `random`, NumPy, torch (CPU + CUDA)."""
    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_probs_indices_past(*_args, **_kwargs):
    """Imported by `stead.py` but never invoked along the paths our wrapper
    runs. If a future call path needs it, implement a past-key-value-aware
    probability lookup here (similar to `sparsamp_utils.get_probs_past`).
    """
    raise NotImplementedError(
        "get_probs_indices_past is a stub. STEAD's encode_diff / decode_diff "
        "do not call it; if something else does, it needs a real implementation."
    )
