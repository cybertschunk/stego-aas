# tests/test_gpt2_recover.py
from django.test import SimpleTestCase
from unittest.mock import patch
import tiktoken

# Adjust this import path to where GPT2Recover and recover_gpt2_tokens live.
from ..decoding import GPT2Recover, recover_gpt2_tokens


def make_validator_for_text(text, fail_first_full=False):
    """
    Build a stateful try_decoding stub bound to 'text' that:
    - Accepts exactly the greedy GPT-2 tokenization for the text.
    - For non-greedy candidates, returns (False, last_ok_byte) where last_ok_byte
      is the UTF-8 byte length of the common byte prefix with the ground truth.
    - If fail_first_full=True, the first call even with the correct greedy sequence
      returns (False, last_ok_byte_before_last_token) to exercise pruning paths,
      then subsequent calls accept the full sequence.
    """
    enc = tiktoken.encoding_for_model("gpt2")
    greedy = enc.encode(text)
    greedy_bytes = b"".join(enc.decode_single_token_bytes(t) for t in greedy)
    last_tok_len = len(enc.decode_single_token_bytes(greedy[-1])) if greedy else 0

    state = {"calls": 0}

    def try_decoding_stub(tokens):
        state["calls"] += 1
        cand_bytes = b"".join(enc.decode_single_token_bytes(t) for t in tokens)

        if tokens == greedy:
            if fail_first_full and state["calls"] == 1 and last_tok_len > 0:
                # Simulate a checkpoint failure just before the last token boundary.
                return False, len(greedy_bytes) - last_tok_len
            return True, len(greedy_bytes)

        # Compute length of common byte prefix to inform pruning.
        pref = 0
        for a, b in zip(cand_bytes, greedy_bytes):
            if a != b:
                break
            pref += 1
        return False, pref

    return try_decoding_stub


class GPT2RecoverTests(SimpleTestCase):
    def setUp(self):
        # Ensure tiktoken is available and correct encoding is used.
        self.enc = tiktoken.encoding_for_model("gpt2")

    def test_simple_ascii_recovery(self):
        text = "This is a test."
        greedy = self.enc.encode(text)

        with patch("myapp.gpt2_recover.try_decoding", make_validator_for_text(text)):
            result = recover_gpt2_tokens(text)

        self.assertEqual(result, greedy)
        self.assertEqual(self.enc.decode(result), text)

    def test_non_ascii_utf8_recovery(self):
        text = "Café 🚀 — naïve\nΔοκιμή 中文测试"
        greedy = self.enc.encode(text)

        with patch("myapp.gpt2_recover.try_decoding", make_validator_for_text(text)):
            result = recover_gpt2_tokens(text)

        self.assertEqual(result, greedy)
        self.assertEqual(self.enc.decode(result), text)

    def test_multiple_spaces_and_leading_space(self):
        text = "  Hello   world!  "
        greedy = self.enc.encode(text)

        with patch("myapp.gpt2_recover.try_decoding", make_validator_for_text(text)):
            result = recover_gpt2_tokens(text)

        self.assertEqual(result, greedy)
        self.assertEqual(self.enc.decode(result), text)

    def test_fail_once_then_constrained_greedy_retry(self):
        text = "Retry once with constrained greedy completion."
        greedy = self.enc.encode(text)

        # First full attempt fails at a byte boundary before the last token; next attempt succeeds.
        with patch("myapp.gpt2_recover.try_decoding", make_validator_for_text(text, fail_first_full=True)):
            result = recover_gpt2_tokens(text)

        self.assertEqual(result, greedy)
        self.assertEqual(self.enc.decode(result), text)

    def test_wrong_edges_pruning_is_exercised(self):
        text = "Trigger pruning via a simulated checkpoint failure."
        validator = make_validator_for_text(text, fail_first_full=True)

        engine = GPT2Recover(text)
        with patch("myapp.gpt2_recover.try_decoding", validator):
            result = engine.recover()

        # After a fail-once path, wrong_edges should record at least one pruned edge.
        self.assertGreater(len(engine.wrong_edges), 0)
        self.assertEqual(self.enc.decode(result), text)

    def test_deterministic_results(self):
        text = "Deterministic behavior should hold across runs."
        validator = make_validator_for_text(text)

        with patch("myapp.gpt2_recover.try_decoding", validator):
            res1 = recover_gpt2_tokens(text)

        # New validator to avoid state leakage.
        with patch("myapp.gpt2_recover.try_decoding", make_validator_for_text(text)):
            res2 = recover_gpt2_tokens(text)

        self.assertEqual(res1, res2)
        self.assertEqual(self.enc.decode(res2), text)

    def test_near_1000_token_limit(self):
        # Build a string that stays under 1000 tokens for GPT-2.
        piece = "hello "
        buf = []
        while True:
            candidate = "".join(buf + [piece])
            if len(self.enc.encode(candidate)) > 950:
                break
            buf.append(piece)
        text = "".join(buf).strip()  # avoid trailing space corner cases

        greedy = self.enc.encode(text)
        self.assertLessEqual(len(greedy), 1000)

        with patch("myapp.gpt2_recover.try_decoding", make_validator_for_text(text)):
            result = recover_gpt2_tokens(text)

        self.assertEqual(result, greedy)
        self.assertEqual(self.enc.decode(result), text)

    def test_roundtrip_bytes_coverage(self):
        text = "Mix: tabs\t, newlines\n, emojis 😊, accents åäö, and punctuation!!!"
        with patch("myapp.gpt2_recover.try_decoding", make_validator_for_text(text)):
            result = recover_gpt2_tokens(text)

        # Ensure the decoded tokens reproduce the exact original text.
        self.assertEqual(self.enc.decode(result), text)
