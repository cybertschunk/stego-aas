"""Reconstructed `config.py` for STEAD.

This file is NOT part of the upstream STEAD release. It is local glue
added to make the vendored STEAD code importable.

Background: the upstream repository
  https://github.com/7-yaya/STEAD @ dc0a48ee7ba98f8e80b047b33105120919bb40a9
ships `stead.py` and `main_stead.py` (by the STEAD authors — Yuang Qi et al.,
USTC) which both `import` from a module called `config`. That `config.py` is
absent from the published repo. This file reconstructs only the *import
surface* — the dataclass and factory shapes the upstream code requires —
inferred from the call sites:

- `stead.py:522,569` — `settings: Settings = Settings()` (no-arg construction)
- `stead.py:525,572` — `algo, temp, top_p, top_k, length, seed = settings()`
  (callable returns this 6-tuple in this order)
- `main_stead.py:80` — `Settings("text", model_name=..., algo=..., top_p=...,
  top_k=..., temp=..., length=...)`
- `main_stead.py:99,113` — `.seed` and `.device` are set externally

Defaults are read from `main_stead.py`'s argparse:
- `top_p=1.0`, `top_k=None`, `temp=1.0`, `length=512`, `seed=42`.

TODO(authors): `stead.py:549` carries a hand comment "# 1.2" next to
`temperature=temp`, suggesting the paper may have used `temp=1.2`. Until
confirmed, this file uses the CLI default `temp=1.0`. The benchmark CSV
records the value in use so any discrepancy with paper numbers is auditable.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Settings:
    """Hyperparameter bundle. Callable returns the 6-tuple stead.py expects."""

    kind: str = "text"
    model_name: Optional[str] = None
    algo: Optional[str] = None
    top_p: float = 1.0
    top_k: Optional[int] = None
    temp: float = 1.0
    length: int = 512
    seed: Optional[int] = None
    device: Any = None

    def __call__(self):
        return (self.algo, self.temp, self.top_p, self.top_k, self.length, self.seed)


def text_default_settings_stead(length: int = 512) -> Settings:
    """Default STEAD configuration: Dream-7B + Stead algo, CLI defaults."""
    return Settings(
        kind="text",
        model_name="Dream-v0-Instruct-7B",
        algo="Stead",
        top_p=1.0,
        top_k=None,
        temp=1.0,
        length=length,
    )


def text_default_settings_sample(length: int = 512) -> Settings:
    """Default config for the `sample` baseline (unused by our benchmark,
    but referenced by `main_stead.py`)."""
    return Settings(
        kind="text",
        model_name="Dream-v0-Instruct-7B",
        algo="sample",
        top_p=1.0,
        top_k=None,
        temp=1.0,
        length=length,
    )
