#!/usr/bin/env python
"""
Quick Performance Test Script

This is a lightweight version of evaluate_backcheck_performance.py for rapid testing.
It runs encode/decode cycles with only 5 random seeds to quickly verify that:
- Performance tracking is working correctly
- The BackCheck algorithm is functioning
- Attempts are being counted and returned properly

Usage:
    python test_evaluation.py

Output: Console output showing attempts per message for 5 test cases
"""

import sys
import os
from collections import Counter

# Add the Django project to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stego-aas', 'stegoaas'))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stegoaas.settings')
import django
django.setup()

from sparsamp_app.encoding import full_encode
from sparsamp_app.decoding import full_decode
from sparsamp_app.model_manager import get_model_manager

# Configuration
HIDDEN_TEXT = "64bitTxt"  # 8 characters = 64 bits
CONTEXT = "Once upon a time in a land far away"
NUM_SEEDS = 5  # Test with just 5 seeds
SEED_START = 10000

print("Testing evaluation script with 5 seeds...")
print(f"Hidden text: '{HIDDEN_TEXT}'")

# Load model
model_manager = get_model_manager()
model_manager.load()
print(f"Model loaded on device: {model_manager.device}")

# Collect attempts
all_attempts = []

for i in range(NUM_SEEDS):
    seed = SEED_START + i
    print(f"\nSeed {seed}:")

    # Encode
    encoded_messages = full_encode(CONTEXT, HIDDEN_TEXT, seed)
    print(f"  Encoded into {len(encoded_messages)} message(s)")

    # Decode
    decoded_text, attempts_list = full_decode(CONTEXT, encoded_messages, seed)
    print(f"  Attempts per message: {attempts_list}")
    print(f"  Decoded: '{decoded_text}' - Match: {decoded_text == HIDDEN_TEXT}")

    all_attempts.extend(attempts_list)

# Statistics
attempt_counts = Counter(all_attempts)
print(f"\nTotal messages: {len(all_attempts)}")
print(f"Attempt distribution: {dict(sorted(attempt_counts.items()))}")
print("\nTest complete!")
