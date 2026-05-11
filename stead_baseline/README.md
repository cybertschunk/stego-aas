# stead_baseline

Vendored copy of **STEAD** (NeurIPS 2025, *Robust Provably Secure Linguistic
Steganography with Diffusion Language Model*) plus the local glue needed to
benchmark it against this repo's BackCheck implementation.

## Attribution

The algorithmic code under `vendor/` is the work of the **STEAD authors**,
not anyone in this repository:

> Yuang Qi, Na Zhao, Qiyi Yao, Benlong Wu, Weiming Zhang, Nenghai Yu,
> Kejiang Chen — Anhui Province Key Laboratory of Digital Security,
> University of Science and Technology of China.
> *"STEAD: Robust Provably Secure Linguistic Steganography with Diffusion
> Language Model."* NeurIPS 2025.

- Upstream: https://github.com/7-yaya/STEAD
- Vendored commit: `dc0a48ee7ba98f8e80b047b33105120919bb40a9` (2025-12-08)
- Files copied from upstream into `vendor/`: `main_stead.py`, `stead.py`,
  `stego_utils.py`, `README.upstream.md` — each carries an attribution
  header pointing back to the original authors.
- Files **added locally** under `vendor/` (NOT by the STEAD authors, just
  glue so their code imports cleanly): `config.py`, `utils.py`. Both files
  declare this in their module docstring.
- One local one-line patch to `vendor/stead.py` (added a missing
  `from collections import Counter` import) — flagged in that file's header.

License: no LICENSE file in upstream repo at the vendored commit. Treat as
"all rights reserved" for redistribution purposes; we use it here for
academic comparison only. **Do not redistribute outside this repo until the
authors publish a license.**

## Status: FUNCTIONAL (with reconstructed config / utils)

The upstream repo is incomplete at the vendored commit. `main_stead.py`
and `stead.py` reference modules absent from the repository:

| Missing import | Symbols referenced | Status here |
|---|---|---|
| `config` | `Settings`, `text_default_settings_stead`, `text_default_settings_sample` | Reconstructed in `vendor/config.py` |
| `utils` | `top_k_logits`, `top_p_logits`, `get_probs_indices_past`, `SingleExampleOutput`, `set_seed` | Reconstructed in `vendor/utils.py`; `get_probs_indices_past` is a NotImplementedError stub (never called along our path) |
| `stega` | `encode_text`, `decode_text` | Not needed for our benchmark — `wrapper.py` calls `stead.encode_text` / `decode_text` directly |
| `random_sample_cy` | `encode_text` | Not needed for our benchmark |

Also patched: `vendor/stead.py` now imports `collections.Counter` (upstream
used it at line 505 without importing it — `find_most_common` would have
NameError'd on first call).

### Caveats / TODOs

- **Hyperparameter defaults** (in `vendor/config.py`) are taken from
  `main_stead.py`'s argparse defaults: `temp=1.0`, `top_p=1.0`,
  `top_k=None`, `length=512`. `stead.py:549` carries a hand comment
  `# 1.2` next to `temperature=temp`, hinting the paper may have used
  `temp=1.2`. Our reported recovery rate is largely temperature-insensitive
  (the encode/decode share the same value), but capacity/perplexity could
  differ. The benchmark CSV records the value used so any discrepancy with
  paper numbers stays auditable.
- **Hardware**: needs a GPU with ≥20 GB VRAM in bfloat16 for
  Dream-v0-Instruct-7B. CPU instantiation will OOM / be infeasible.
- When the authors publish official `config.py` / `utils.py`, drop them
  into `vendor/` and these reconstructions become moot.
