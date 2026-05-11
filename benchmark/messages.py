"""Deterministic 512-bit message generation for the benchmark.

Both BackCheck and STEAD must operate on identical bit content. BackCheck's
pipeline reconstructs strings through UTF-8 and strips trailing NUL bytes,
so we draw from an alphabet that is guaranteed to be single-byte UTF-8 and
contains no zero byte.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    seed: int
    ascii_str: str   # for BackCheck (passed as plaintext)
    bits: str        # for STEAD (passed as binary string)

    @property
    def bit_length(self) -> int:
        return len(self.bits)


_ALPHABET = string.ascii_letters + string.digits  # 62 chars, single-byte UTF-8, no NUL


def generate_message(seed: int, num_chars: int = 64) -> Message:
    """Generate a deterministic message with `num_chars * 8` bits.

    Default `num_chars=64` yields exactly 512 bits.
    """
    rng = random.Random(seed)
    ascii_str = "".join(rng.choice(_ALPHABET) for _ in range(num_chars))
    bits = "".join(f"{b:08b}" for b in ascii_str.encode("ascii"))
    return Message(seed=seed, ascii_str=ascii_str, bits=bits)
