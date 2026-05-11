"""
Thin adapter that wraps STEAD's `encode_text` / `decode_text` for the
benchmark harness.

`config.py` and `utils.py` were absent from the upstream repo at the
vendored commit; they are reconstructed under `stead_baseline/vendor/`
from the surface area visible in the vendored `stead.py` and
`main_stead.py`. See `stead_baseline/README.md` for the limitations.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import Optional


class STEADUnavailable(RuntimeError):
    """Raised when STEAD cannot run because upstream modules are missing."""


@dataclass
class SteadEncodeOutput:
    stego_text: str
    total_capacity: int
    embed_time: float
    elapsed_s: float


@dataclass
class SteadDecodeOutput:
    recovered_bits: Optional[str]
    elapsed_s: float


class SteadAdapter:
    """
    Loads Dream-v0-Instruct-7B once, then encodes/decodes message bits.

    Refuses to construct while `config.py` / `utils.py` are stubs.
    """

    name = "stead"
    DEFAULT_MODEL = "Dream-org/Dream-v0-Instruct-7B"

    def __init__(self, device: Optional[str] = None, length: int = 512):
        self._verify_upstream_complete()

        import torch
        from transformers import AutoModel, AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.length = length

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.DEFAULT_MODEL, trust_remote_code=True
        )
        self._model = (
            AutoModel.from_pretrained(
                self.DEFAULT_MODEL,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def encode(self, bits: str, prompt: str, seed: int) -> SteadEncodeOutput:
        from stead_baseline.vendor.stead import encode_text
        from stead_baseline.wrapper import _build_settings  # noqa: WPS433
        from stead_baseline.wrapper import _set_seed        # noqa: WPS433

        settings = _build_settings(length=self.length, seed=seed, device=self.device)
        _set_seed(seed)

        t0 = time.time()
        single_output, embed_time = encode_text(
            self._model,
            self._tokenizer,
            message_bits=bits,
            prompt=prompt,
            settings=settings,
        )
        elapsed = time.time() - t0
        return SteadEncodeOutput(
            stego_text=single_output.stego_object,
            total_capacity=int(getattr(single_output, "n_bits", 0)),
            embed_time=embed_time,
            elapsed_s=elapsed,
        )

    def decode(self, stego_text: str, prompt: str, seed: int,
               expected_bits: int = 512) -> SteadDecodeOutput:
        import torch
        from stead_baseline.vendor.stead import decode_text
        from stead_baseline.wrapper import _build_settings, _set_seed  # noqa: WPS433

        stego_ids = self._tokenizer(stego_text, return_tensors="pt")["input_ids"][0].tolist()
        # Strip BOS if the tokenizer added one
        if stego_ids and stego_ids[0] == self._tokenizer.bos_token_id:
            stego_ids = stego_ids[1:]

        settings = _build_settings(length=self.length, seed=seed, device=self.device)
        _set_seed(seed)

        t0 = time.time()
        decoded = decode_text(
            self._model,
            self._tokenizer,
            stego=stego_ids,
            prompt=prompt,
            settings=settings,
        )
        elapsed = time.time() - t0

        # STEAD may emit 'x' for unresolved bit positions or terminate early.
        recovered: Optional[str]
        if decoded is None or "x" in decoded:
            recovered = None
        else:
            recovered = decoded[:expected_bits]
            if len(recovered) < expected_bits:
                recovered = None  # short decode == failure for exact-match
        return SteadDecodeOutput(recovered_bits=recovered, elapsed_s=elapsed)

    # ------------------------------------------------------------------ #
    # Guards                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _verify_upstream_complete() -> None:
        # Importing `stead_baseline` puts vendor/ on sys.path so that bare
        # `from config import ...` and `from utils import ...` inside the
        # vendored files resolve to our reconstructed modules.
        import stead_baseline  # noqa: F401
        _OWN = {"config", "utils"}
        for name in _OWN:
            try:
                importlib.import_module(name)
            except ModuleNotFoundError as exc:
                missing = exc.name or name
                if missing in _OWN:
                    raise STEADUnavailable(
                        f"Reconstructed module {missing!r} missing from "
                        "stead_baseline/vendor/."
                    ) from exc
                raise STEADUnavailable(
                    f"Third-party dependency {missing!r} is not installed in this "
                    "environment. STEAD needs scipy, tqdm, transformers, torch "
                    "(see stead_baseline/README.md)."
                ) from exc


# Helpers deferred until upstream lands. These imports execute only after
# `_verify_upstream_complete` passes, so the stubs never get exercised here.

def _build_settings(*, length: int, seed: int, device):
    # Bare imports so vendored stead.py and this wrapper share the same
    # Settings class (stead.py also does `from config import Settings`).
    import stead_baseline  # noqa: F401 — ensures vendor/ is on sys.path
    from config import text_default_settings_stead  # type: ignore
    try:
        settings = text_default_settings_stead(length=length)
    except TypeError:
        # Tolerate a module-level Settings instance if upstream later switches.
        settings = text_default_settings_stead  # type: ignore[assignment]
    settings.seed = seed
    settings.device = device
    settings.length = length
    return settings


def _set_seed(seed: int) -> None:
    import stead_baseline  # noqa: F401
    from utils import set_seed  # type: ignore
    set_seed(seed)
