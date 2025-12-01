#!/usr/bin/env python
"""
BackCheck Algorithm Performance Evaluation Script

This script provides comprehensive performance benchmarking of the BackCheck algorithm,
which resolves tokenization ambiguity during steganographic message decoding.

What it does:
1. Encodes a 1024-bit (128 character) hidden message using 100 different random seeds
2. Decodes each encoded message using the BackCheck algorithm
3. Tracks the number of BackCheck attempts required for each message segment
4. Records encoding/decoding timing information
5. Outputs detailed statistics and exports results to CSV

Performance Metrics Tracked:
- Number of BackCheck attempts per message
- Success/failure rates
- Encoding and decoding timing
- Attempt distribution across all messages

Output CSV Format:
- Header rows with configuration parameters (text length, TOP_P, MAX_BACKCHECK_ATTEMPTS)
- Data rows: attempt_count, message_count
- Example: "4,13" means 13 messages required exactly 4 attempts to decode
- Failed decodings counted as MAX_BACKCHECK_ATTEMPTS + 1

Usage:
    python evaluate_backcheck_performance.py

Expected Runtime:
- CPU: ~30-60 minutes for 100 seeds
- GPU: ~10-20 minutes for 100 seeds

Output Files:
- backcheck_performance_YYYYMMDD_HHMMSS.csv (timestamped results)
"""

import sys
import os
from collections import Counter
import csv
from datetime import datetime
import time
import random as py_random

# Add the Django project to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stego-aas', 'stegoaas'))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stegoaas.settings')
import django
django.setup()

from sparsamp_app.encoding import full_encode
from sparsamp_app.decoding import full_decode
from sparsamp_app.model_manager import get_model_manager
from sparsamp_app.constants import MAX_BACKCHECK_ATTEMPTS, TOP_P

def run_performance_evaluation():
    """
    Run performance evaluation of BackCheck algorithm
    """
    # Configuration
    HIDDEN_TEXT = "This is exactly 1024 bits of hidden text for testing the BackCheck algorithm performance with messages in steganography research"  # 128 characters = 1024 bits
    assert len(HIDDEN_TEXT) == 128, f"Hidden text must be exactly 128 characters (1024 bits), got {len(HIDDEN_TEXT)}"
    CONTEXT = "Once upon a time in a land far away"
    NUM_SEEDS = 100
    SEED_MIN = 10000
    SEED_MAX = 999999  # Random seeds in this range

    print("=" * 70)
    print("BackCheck Algorithm Performance Evaluation")
    print("=" * 70)
    print(f"Hidden text: '{HIDDEN_TEXT}' ({len(HIDDEN_TEXT)} characters)")
    print(f"Context: '{CONTEXT}'")
    print(f"Number of seeds to test: {NUM_SEEDS}")
    print(f"Seed range: {SEED_MIN} to {SEED_MAX} (random)")
    print("=" * 70)

    # Generate random seeds
    py_random.seed(42)  # For reproducibility
    test_seeds = [py_random.randint(SEED_MIN, SEED_MAX) for _ in range(NUM_SEEDS)]

    # Load model once at the beginning
    print("\nLoading model...")
    model_manager = get_model_manager()
    model_manager.load()
    print(f"Model loaded on device: {model_manager.device}")

    # Collect all attempts per message across all seeds
    all_attempts = []
    failed_messages = 0  # Track messages that failed (exceeded MAX_BACKCHECK_ATTEMPTS)
    encode_times = []
    decode_times = []

    print(f"\nRunning {NUM_SEEDS} encode/decode cycles...")
    print("-" * 70)

    start_time = time.time()

    for i in range(NUM_SEEDS):
        seed = test_seeds[i]

        # Progress indicator - show every iteration now
        elapsed = time.time() - start_time
        avg_time_per_seed = elapsed / (i + 1) if i > 0 else 0
        remaining_seeds = NUM_SEEDS - (i + 1)
        estimated_remaining = avg_time_per_seed * remaining_seeds

        print(f"\n[{i + 1}/{NUM_SEEDS}] Seed: {seed} | Elapsed: {elapsed:.1f}s | ETA: {estimated_remaining:.1f}s")

        try:
            # Encode the hidden text
            encode_start = time.time()
            encoded_messages = full_encode(
                context=CONTEXT,
                message_text=HIDDEN_TEXT,
                random_seed=seed
            )
            encode_time = time.time() - encode_start
            encode_times.append(encode_time)

            print(f"  Encode: {encode_time:.3f}s → {len(encoded_messages)} message(s)")

            # Decode the messages
            decode_start = time.time()
            decoded_text, attempts_list = full_decode(
                context=CONTEXT,
                messages=encoded_messages,
                random_seed=seed
            )
            decode_time = time.time() - decode_start
            decode_times.append(decode_time)

            print(f"  Decode: {decode_time:.3f}s → Attempts: {attempts_list}")

            # Verify decoding was successful
            if decoded_text != HIDDEN_TEXT:
                print(f"  ⚠ WARNING: Decoded text doesn't match!")
                print(f"    Expected: '{HIDDEN_TEXT}'")
                print(f"    Got: '{decoded_text}'")
                # Count each failed message as requiring more than MAX_BACKCHECK_ATTEMPTS
                failed_messages += len(encoded_messages)
                continue

            # Add all message attempts to our collection
            all_attempts.extend(attempts_list)
            print(f"  ✓ Success")

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            # If we know how many messages were encoded, count them as failed
            try:
                if 'encoded_messages' in locals():
                    failed_messages += len(encoded_messages)
            except:
                pass
            continue

    total_elapsed = time.time() - start_time

    print(f"\n{'=' * 70}")
    print(f"Completed {NUM_SEEDS} encode/decode cycles in {total_elapsed:.1f}s")
    print(f"{'=' * 70}")

    # Count occurrences of each attempt value
    attempt_counts = Counter(all_attempts)

    # Add failed messages as requiring more than MAX_BACKCHECK_ATTEMPTS
    # We'll use MAX_BACKCHECK_ATTEMPTS + 1 to represent "failed after max attempts"
    if failed_messages > 0:
        attempt_counts[MAX_BACKCHECK_ATTEMPTS + 1] = failed_messages

    # Generate statistics
    total_messages = len(all_attempts) + failed_messages
    successful_messages = len(all_attempts)

    print(f"\nMessage Statistics:")
    print(f"  Total messages processed: {total_messages}")
    print(f"  Successfully decoded: {successful_messages}")
    print(f"  Failed (>MAX_ATTEMPTS): {failed_messages}")
    print(f"  Success rate: {(successful_messages / total_messages * 100):.2f}%")
    print(f"  Average messages per seed: {total_messages / NUM_SEEDS:.2f}")
    print(f"  Unique attempt counts: {len(attempt_counts)}")

    if all_attempts:
        print(f"  Min attempts (successful): {min(all_attempts)}")
        print(f"  Max attempts (successful): {max(all_attempts)}")
        print(f"  Average attempts (successful): {sum(all_attempts) / successful_messages:.2f}")
        print(f"  Average attempts (all): {(sum(all_attempts) + failed_messages * (MAX_BACKCHECK_ATTEMPTS + 1)) / total_messages:.2f}")

    print(f"\nTiming Statistics:")
    if encode_times:
        print(f"  Encoding:")
        print(f"    Average time: {sum(encode_times) / len(encode_times):.3f}s")
        print(f"    Min time: {min(encode_times):.3f}s")
        print(f"    Max time: {max(encode_times):.3f}s")
        print(f"    Total time: {sum(encode_times):.1f}s")

    if decode_times:
        print(f"  Decoding:")
        print(f"    Average time: {sum(decode_times) / len(decode_times):.3f}s")
        print(f"    Min time: {min(decode_times):.3f}s")
        print(f"    Max time: {max(decode_times):.3f}s")
        print(f"    Total time: {sum(decode_times):.1f}s")

    # Display attempt distribution
    print(f"\nAttempt Distribution:")
    print(f"  {'Attempts':<20} {'Messages':<12} {'Percentage'}")
    print(f"  {'-'*20} {'-'*12} {'-'*12}")

    for attempts in sorted(attempt_counts.keys()):
        count = attempt_counts[attempts]
        percentage = (count / total_messages) * 100
        if attempts == MAX_BACKCHECK_ATTEMPTS + 1:
            attempts_str = f">MAX ({attempts})"
        else:
            attempts_str = str(attempts)
        print(f"  {attempts_str:<20} {count:<12} {percentage:>6.2f}%")

    # Write results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"backcheck_performance_{timestamp}.csv"

    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header with explanation
        writer.writerow(['attempt', 'count'])
        writer.writerow([f'# Hidden text length: {len(HIDDEN_TEXT)} characters = {len(HIDDEN_TEXT) * 8} bits'])
        writer.writerow([f'# TOP_P value: {TOP_P}'])
        writer.writerow([f'# MAX_BACKCHECK_ATTEMPTS = {MAX_BACKCHECK_ATTEMPTS}'])
        writer.writerow([f'# Failed messages are counted as {MAX_BACKCHECK_ATTEMPTS + 1} (>MAX_ATTEMPTS)'])

        # Write data sorted by attempt count
        for attempts in sorted(attempt_counts.keys()):
            count = attempt_counts[attempts]
            writer.writerow([attempts, count])

    print(f"\nResults written to: {csv_filename}")
    print(f"Note: Failed decodings are counted as attempt count {MAX_BACKCHECK_ATTEMPTS + 1} (>MAX_ATTEMPTS)")
    print("=" * 70)
    print("Evaluation complete!")
    print("=" * 70)

if __name__ == "__main__":
    try:
        run_performance_evaluation()
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
