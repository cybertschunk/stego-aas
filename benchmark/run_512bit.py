#!/usr/bin/env python
"""512-bit message recovery benchmark: BackCheck vs. STEAD.

Encodes the same 512-bit message with each system under N random seeds and
records exact-match recovery, encode/decode timings, and stego text length.

Usage:
    python benchmark/run_512bit.py --num-seeds 50
    python benchmark/run_512bit.py --num-seeds 5 --skip-stead  # baseline only

Output: `benchmark_512bit_YYYYMMDD_HHMMSS.csv` in the current directory.
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Iterable

# Allow running as `python benchmark/run_512bit.py` from the repo root.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark.adapters import (  # noqa: E402
    BackCheckAdapter,
    DecodeResult,
    EncodeResult,
    StegoAdapter,
    make_stead_adapter,
)
from benchmark.messages import generate_message  # noqa: E402


DEFAULT_PROMPT = "Once upon a time in a land far away"
SEED_MIN = 10_000
SEED_MAX = 999_999
META_SEED = 42  # determines the test-seed sequence


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--num-seeds", type=int, default=50,
                   help="Number of (seed, system) trials per system (default: 50)")
    p.add_argument("--prompt", default=DEFAULT_PROMPT,
                   help="Shared prompt/context fed to both systems")
    p.add_argument("--num-chars", type=int, default=64,
                   help="Message length in chars (64 = 512 bits)")
    p.add_argument("--skip-stead", action="store_true",
                   help="Run BackCheck only (STEAD's upstream code may be missing)")
    p.add_argument("--out", default=None,
                   help="Output CSV path (default: ./benchmark_512bit_<timestamp>.csv)")
    return p.parse_args()


def _run_one(adapter: StegoAdapter, *, bits: str, prompt: str, seed: int):
    encode_err: str | None = None
    decode_err: str | None = None
    enc: EncodeResult | None = None
    dec: DecodeResult | None = None
    try:
        enc = adapter.encode(bits, prompt, seed)
    except Exception as exc:  # noqa: BLE001
        encode_err = repr(exc)
        return enc, dec, encode_err, decode_err

    try:
        dec = adapter.decode(enc.stego_text, prompt, seed)
    except Exception as exc:  # noqa: BLE001
        decode_err = repr(exc)

    return enc, dec, encode_err, decode_err


def _summarize(rows: list[dict]) -> None:
    by_system: dict[str, list[dict]] = {}
    for row in rows:
        by_system.setdefault(row["system"], []).append(row)

    print()
    print("=" * 78)
    print(f"{'System':<12} {'Trials':>7} {'Success':>9} {'Rate':>7}  "
          f"{'Encode (s)':>11} {'Decode (s)':>11} {'Stego chars':>11}")
    print("-" * 78)
    for system, srows in by_system.items():
        succ = sum(1 for r in srows if r["success"] == "1")
        total = len(srows)
        rate = (succ / total * 100) if total else 0.0
        enc_times = [float(r["encode_s"]) for r in srows if r["encode_s"]]
        dec_times = [float(r["decode_s"]) for r in srows if r["decode_s"]]
        stego_lens = [int(r["stego_chars"]) for r in srows if r["stego_chars"]]

        print(
            f"{system:<12} {total:>7} {succ:>9} {rate:>6.1f}%  "
            f"{statistics.mean(enc_times) if enc_times else 0:>11.3f} "
            f"{statistics.mean(dec_times) if dec_times else 0:>11.3f} "
            f"{int(statistics.mean(stego_lens)) if stego_lens else 0:>11}"
        )
    print("=" * 78)


def main() -> int:
    args = _parse_args()

    # Reproducible seed sequence: same META_SEED → same test seeds across runs.
    rng = random.Random(META_SEED)
    test_seeds = [rng.randint(SEED_MIN, SEED_MAX) for _ in range(args.num_seeds)]

    print(f"=== 512-bit recovery benchmark — {args.num_seeds} seed(s) ===")
    print(f"prompt           : {args.prompt!r}")
    print(f"message chars    : {args.num_chars}  (bits = {args.num_chars * 8})")
    print(f"skip stead       : {args.skip_stead}")
    print()

    print("Loading BackCheck adapter…")
    backcheck = BackCheckAdapter()
    adapters: list[StegoAdapter] = [backcheck]

    if not args.skip_stead:
        try:
            print("Loading STEAD adapter (Dream-7B)…")
            adapters.append(make_stead_adapter())
        except Exception as exc:  # noqa: BLE001
            print(f"!! STEAD unavailable: {exc}", file=sys.stderr)
            print("!! Continuing with BackCheck only. Re-run after upstream is complete.",
                  file=sys.stderr)

    rows: list[dict] = []
    overall_start = time.time()

    for i, seed in enumerate(test_seeds, start=1):
        msg = generate_message(seed=seed, num_chars=args.num_chars)
        elapsed = time.time() - overall_start
        avg = elapsed / max(i - 1, 1) if i > 1 else 0
        eta = avg * (args.num_seeds - i + 1)
        print(f"\n[{i}/{args.num_seeds}] seed={seed} | elapsed={elapsed:.0f}s | eta~{eta:.0f}s")

        for adapter in adapters:
            enc, dec, enc_err, dec_err = _run_one(
                adapter, bits=msg.bits, prompt=args.prompt, seed=seed,
            )

            success = (
                enc is not None
                and dec is not None
                and dec.recovered_bits is not None
                and dec.recovered_bits == msg.bits
            )

            notes = []
            if enc_err:
                notes.append(f"encode_error={enc_err}")
            if dec_err:
                notes.append(f"decode_error={dec_err}")
            if enc is not None:
                extra = ", ".join(f"{k}={v}" for k, v in enc.extra.items())
                if extra:
                    notes.append(f"enc[{extra}]")
            if dec is not None and dec.extra:
                # Drop the full decoded_text from CSV to keep it tidy
                tidy = {k: v for k, v in dec.extra.items() if k != "decoded_text"}
                if tidy:
                    notes.append(f"dec[{', '.join(f'{k}={v}' for k, v in tidy.items())}]")

            row = {
                "seed": seed,
                "system": adapter.name,
                "success": "1" if success else "0",
                "encode_s": f"{enc.elapsed_s:.4f}" if enc else "",
                "decode_s": f"{dec.elapsed_s:.4f}" if dec else "",
                "stego_chars": str(len(enc.stego_text)) if enc else "",
                "notes": " | ".join(notes),
            }
            rows.append(row)

            ok = "OK " if success else "FAIL"
            print(f"  {adapter.name:<10} {ok}  enc={row['encode_s']}s "
                  f"dec={row['decode_s']}s  chars={row['stego_chars']}")

    # Write CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else Path(f"benchmark_512bit_{timestamp}.csv")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["seed", "system", "success", "encode_s", "decode_s",
                           "stego_chars", "notes"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nWrote {len(rows)} rows to {out_path}")
    _summarize(rows)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
