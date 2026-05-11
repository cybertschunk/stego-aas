#!/usr/bin/env python
"""Run STEAD on a paper-faithful single example to verify it works.

Mirrors the per-example logic of `main_stead.py::test_dataset` (vendored
under `stead_baseline/vendor/`) without the IMDB dataset dependency or
the full 200-sample loop. Use this to confirm STEAD's encode/decode
pipeline is healthy on Dream-7B before doing anything fancy.

Run on a GPU box (≥20 GB VRAM):

    cd <repo-root>
    python -m stead_baseline.reproduce_paper          # 1 example, defaults
    python -m stead_baseline.reproduce_paper --n 5    # 5 examples
    python -m stead_baseline.reproduce_paper --seed 7 --length 512 --temp 1.0

Reports per-example correctness (bit accuracy over `total_capacity` bits),
total capacity, embedding rate, perplexity, encode/decode wall time. No
hyperparameters of the published algorithm are altered; we use the
defaults baked into `vendor/config.text_default_settings_stead`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Make `from config import …` etc. resolve to vendor/.
import stead_baseline  # noqa: F401

from config import text_default_settings_stead  # type: ignore
from utils import set_seed                       # type: ignore
from stead import encode_text, decode_text        # type: ignore


DREAM_MODEL = "Dream-org/Dream-v0-Instruct-7B"

# A short IMDB-style prompt (the paper feeds the first two sentences of an
# IMDB review as `stego_prompt`; we use a representative example so this
# script has no external data dependency).
DEFAULT_PROMPT = (
    "I went into this movie with low expectations and was pleasantly surprised. "
    "The acting was solid and the cinematography genuinely beautiful."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=1,
                   help="Number of examples to run (default 1).")
    p.add_argument("--seed", type=int, default=42,
                   help="Base seed; per-example seed is base+i (default 42).")
    p.add_argument("--message-bits", type=int, default=3000,
                   help="Length of the binary message to embed per example "
                        "(default 3000, matching main_stead.py).")
    p.add_argument("--prompt", default=DEFAULT_PROMPT,
                   help="Prompt fed to Dream-7B (default: a short IMDB-style review).")
    p.add_argument("--length", type=int, default=None,
                   help="Override settings.length (max_new_tokens). Default = paper default.")
    p.add_argument("--temp", type=float, default=None,
                   help="Override settings.temp. Default = paper default (1.0).")
    p.add_argument("--out", default=None,
                   help="Where to write the per-example JSON (default: reproduce_paper_<ts>.json)")
    return p.parse_args()


def load_dream(device: torch.device):
    print(f"Loading {DREAM_MODEL} on {device} (bfloat16)…", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(DREAM_MODEL, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(
            DREAM_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        .to(device)
        .eval()
    )
    return model, tokenizer


def random_bits(rng: random.Random, n: int) -> str:
    """Generate `n` random '0'/'1' chars deterministically from `rng`."""
    return "".join(rng.choice("01") for _ in range(n))


def run_one(model, tokenizer, *, prompt: str, message_bits: str,
            settings, seed: int) -> dict:
    """One encode → decode round-trip, returning the same fields the paper
    test loop records (subset)."""
    settings.seed = seed

    t0 = time.time()
    single_output, embed_time = encode_text(
        model, tokenizer,
        message_bits=message_bits,
        prompt=prompt,
        settings=settings,
    )
    t_enc = time.time() - t0

    stego_text = single_output.stego_object
    total_capacity = single_output.n_bits

    # Re-tokenize the stego string so we measure end-to-end recoverability
    # (matches the paper's evaluator flow).
    stego_ids = tokenizer(stego_text, return_tensors="pt")["input_ids"][0].tolist()
    if stego_ids and stego_ids[0] == tokenizer.bos_token_id:
        stego_ids = stego_ids[1:]

    t0 = time.time()
    decoded = decode_text(model, tokenizer, stego_ids, prompt, settings)
    t_dec = time.time() - t0

    # Correctness: fraction of the embedded portion that decoded matches.
    if total_capacity == 0:
        correctness = float("nan")
    elif decoded is None or len(decoded) < total_capacity:
        correctness = 0.0
    else:
        emb = np.array([int(b) for b in message_bits[:total_capacity]])
        dec = np.array([int(b) if b in "01" else -1 for b in decoded[:total_capacity]])
        correctness = float((emb == dec).mean())

    return {
        "seed": seed,
        "prompt": prompt,
        "stego_text": stego_text,
        "total_capacity": int(total_capacity),
        "embedding_rate": float(getattr(single_output, "embedding_rate", 0.0)),
        "perplexity": float(getattr(single_output, "perplexity", float("nan"))),
        "ave_kld": float(getattr(single_output, "ave_kld", float("nan"))),
        "n_tokens": int(getattr(single_output, "n_tokens", 0)),
        "embed_time_s": float(embed_time),
        "encode_total_s": float(t_enc),
        "decode_total_s": float(t_dec),
        "correctness": correctness,
        "decoded_len": 0 if decoded is None else len(decoded),
    }


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        print("FATAL: no CUDA device. STEAD needs ~20 GB GPU VRAM.", file=sys.stderr)
        return 2
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    set_seed(args.seed)
    model, tokenizer = load_dream(device)
    print(f"Model loaded.  Param count: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    settings = text_default_settings_stead(length=args.length if args.length else 512)
    settings.device = device
    if args.temp is not None:
        settings.temp = args.temp
    print(f"Settings: algo={settings.algo}  length={settings.length}  "
          f"temp={settings.temp}  top_p={settings.top_p}  top_k={settings.top_k}")
    print()

    rng = random.Random(args.seed)
    results = []
    for i in range(args.n):
        bits = random_bits(rng, args.message_bits)
        seed_i = args.seed + i
        print(f"--- example {i + 1}/{args.n}  (seed={seed_i}, message_bits={args.message_bits}) ---")
        try:
            r = run_one(
                model, tokenizer,
                prompt=args.prompt,
                message_bits=bits,
                settings=settings,
                seed=seed_i,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  RAISED: {type(exc).__name__}: {exc}")
            results.append({"seed": seed_i, "error": repr(exc)})
            continue

        print(f"  capacity={r['total_capacity']} bits  "
              f"embed_rate={r['embedding_rate']:.3f} bits/token  "
              f"correctness={r['correctness']:.4f}")
        print(f"  encode={r['encode_total_s']:.1f}s  decode={r['decode_total_s']:.1f}s  "
              f"perplexity={r['perplexity']:.2f}")
        print(f"  stego[:160]={r['stego_text'][:160]!r}")
        results.append(r)

    # Summary
    ok = [r for r in results if r.get("correctness", 0) >= 0.999]
    print()
    print("=" * 70)
    print(f"Examples completed:        {len(results)}")
    print(f"Bit-perfect (>=99.9%):     {len(ok)}")
    if results:
        caps = [r.get("total_capacity", 0) for r in results if "total_capacity" in r]
        corrs = [r["correctness"] for r in results if "correctness" in r and r["correctness"] == r["correctness"]]
        if caps:
            print(f"Mean total_capacity:       {sum(caps)/len(caps):.0f} bits")
        if corrs:
            print(f"Mean correctness:          {sum(corrs)/len(corrs):.4f}")
    print("=" * 70)

    out = args.out or f"reproduce_paper_{int(time.time())}.json"
    Path(out).write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
