"""Unified adapter interface for stego systems under benchmark.

Each adapter exposes the same `encode(bits, prompt, seed)` and
`decode(stego, prompt, seed)` surface, so the runner can A/B them with
identical inputs.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


# ---- Django bootstrap (needed before importing sparsamp_app) -------------- #

def _ensure_django() -> None:
    if os.environ.get("DJANGO_SETTINGS_MODULE"):
        return
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, os.pardir))
    django_root = os.path.join(project_root, "stego-aas", "stegoaas")
    if django_root not in sys.path:
        sys.path.insert(0, django_root)
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "stegoaas.settings")
    import django
    django.setup()


# ---- Shared types --------------------------------------------------------- #

@dataclass
class EncodeResult:
    stego_text: str
    elapsed_s: float
    extra: dict = field(default_factory=dict)


@dataclass
class DecodeResult:
    recovered_bits: Optional[str]
    elapsed_s: float
    extra: dict = field(default_factory=dict)


class StegoAdapter(Protocol):
    name: str

    def encode(self, bits: str, prompt: str, seed: int) -> EncodeResult: ...
    def decode(self, stego: str, prompt: str, seed: int,
               expected_bits: int) -> DecodeResult: ...


# ---- BackCheck adapter ---------------------------------------------------- #

# Sentinel inserted between BackCheck's per-message stego strings so we can
# round-trip the list through a single CSV "stego_text" column.
_BACKCHECK_SEP = "\x1f"  # ASCII Unit Separator — won't appear in GPT-2 output


class BackCheckAdapter:
    """Adapter around `sparsamp_app.encoding.full_encode` / `full_decode`."""

    name = "backcheck"

    def __init__(self) -> None:
        _ensure_django()
        # Imported here so Django setup runs before module import.
        from sparsamp_app.encoding import full_encode  # noqa: WPS433
        from sparsamp_app.decoding import full_decode  # noqa: WPS433
        from sparsamp_app.model_manager import get_model_manager  # noqa: WPS433
        from sparsamp_app.sparsamp_utils import (  # noqa: WPS433
            string_to_utf8_binary,
            utf8_binary_to_string,
        )

        self._full_encode = full_encode
        self._full_decode = full_decode
        self._string_to_bits = string_to_utf8_binary
        self._bits_to_string = utf8_binary_to_string

        # Preload GPT-2 once so the first timed call isn't measuring model
        # download / load time.
        get_model_manager().load()

    def encode(self, bits: str, prompt: str, seed: int) -> EncodeResult:
        if len(bits) % 8 != 0:
            raise ValueError(f"BackCheck needs a byte-aligned bit string, got len={len(bits)}")
        plaintext = self._bits_to_string(bits)

        t0 = time.time()
        messages = self._full_encode(
            context=prompt,
            message_text=plaintext,
            random_seed=seed,
        )
        elapsed = time.time() - t0
        return EncodeResult(
            stego_text=_BACKCHECK_SEP.join(messages),
            elapsed_s=elapsed,
            extra={"num_messages": len(messages)},
        )

    def decode(self, stego: str, prompt: str, seed: int,
               expected_bits: int) -> DecodeResult:
        messages = stego.split(_BACKCHECK_SEP)

        t0 = time.time()
        try:
            decoded_text, attempts = self._full_decode(
                context=prompt,
                messages=messages,
                random_seed=seed,
            )
        except Exception as exc:  # noqa: BLE001 — surface as soft failure
            return DecodeResult(
                recovered_bits=None,
                elapsed_s=time.time() - t0,
                extra={"error": repr(exc), "attempts": []},
            )
        elapsed = time.time() - t0

        try:
            recovered_bits: Optional[str] = self._string_to_bits(decoded_text)
        except UnicodeDecodeError:
            recovered_bits = None
        # BackCheck ignores expected_bits (its decode returns exactly what was
        # encoded). The param is part of the protocol for STEAD's truncation.
        del expected_bits

        return DecodeResult(
            recovered_bits=recovered_bits,
            elapsed_s=elapsed,
            extra={"attempts": list(attempts), "decoded_text": decoded_text},
        )


# ---- STEAD adapter (lazy facade) ------------------------------------------ #

def make_stead_adapter(**kwargs: Any) -> StegoAdapter:
    """Construct the STEAD adapter. Raises STEADUnavailable until upstream lands."""
    from stead_baseline.wrapper import SteadAdapter
    inner = SteadAdapter(**kwargs)

    class _SteadShim:
        name = "stead"

        def encode(self, bits: str, prompt: str, seed: int) -> EncodeResult:
            out = inner.encode(bits, prompt, seed)
            return EncodeResult(
                stego_text=out.stego_text,
                elapsed_s=out.elapsed_s,
                extra={"total_capacity": out.total_capacity, "embed_time": out.embed_time},
            )

        def decode(self, stego: str, prompt: str, seed: int,
                   expected_bits: int) -> DecodeResult:
            out = inner.decode(stego, prompt, seed, expected_bits=expected_bits)
            return DecodeResult(
                recovered_bits=out.recovered_bits,
                elapsed_s=out.elapsed_s,
                extra={},
            )

    return _SteadShim()
