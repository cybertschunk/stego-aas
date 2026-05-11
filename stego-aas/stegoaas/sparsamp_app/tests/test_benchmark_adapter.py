"""End-to-end round-trip test for the BackCheckAdapter.

Verifies that bits → str → encode → decode → str → bits is an identity on
a deterministic short message. Catches regressions in the adapter's
bit/byte conversion glue without depending on STEAD.
"""

import os
import sys

from django.test import TestCase

# Ensure the benchmark/ package (at repo root) is importable when this test
# runs via `manage.py test`.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class BackCheckAdapterRoundTripTest(TestCase):
    """The adapter must faithfully recover the bits it was given."""

    def test_short_message_round_trip(self):
        from benchmark.adapters import BackCheckAdapter
        from benchmark.messages import generate_message

        adapter = BackCheckAdapter()
        # 8 chars = 64 bits — short enough for fast test, real bytes-aligned.
        msg = generate_message(seed=12345, num_chars=8)

        enc = adapter.encode(msg.bits, prompt="Once upon a time", seed=12345)
        self.assertGreater(len(enc.stego_text), 0)

        dec = adapter.decode(enc.stego_text, prompt="Once upon a time", seed=12345)
        self.assertEqual(
            dec.recovered_bits, msg.bits,
            f"Adapter round-trip lost bits. "
            f"decoded_text={dec.extra.get('decoded_text')!r}, "
            f"attempts={dec.extra.get('attempts')}",
        )
