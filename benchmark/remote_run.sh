#!/usr/bin/env bash
# Provision deps + run smoke (N=3), then full benchmark (N=50) if smoke OK.
# Designed to be launched on the vast.ai instance via:
#   nohup bash remote_run.sh > /workspace/run.log 2>&1 &
# Writes a /workspace/DONE marker (with exit reason) on completion.

set -u
set -o pipefail
exec 2>&1   # merge stderr into our pipe so everything is logged

REPO=/workspace/stego-aas
HF=/workspace/hf-cache
OUT=/workspace/results
mkdir -p "$OUT"
export HF_HOME="$HF"
export PYTHONUNBUFFERED=1   # critical: streaming output for tqdm + prints

cd "$REPO"

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log()   { echo "[$(stamp)] $*"; }
mark_done() { echo "$1" > /workspace/DONE; log "DONE: $1"; exit "${2:-0}"; }

trap 'mark_done "trap-killed-on-signal" 130' INT TERM

# ----------------------------------------------------------------------------
log "=== Step 1: install requirements ==="
pip install -q -r requirements.txt || mark_done "pip-base-failed" 1
pip install -q -r requirements-stead.txt || mark_done "pip-stead-failed" 1
log "pip installs done"

# ----------------------------------------------------------------------------
log "=== Step 2: CUDA + import sanity ==="
python <<'PY' || mark_done "cuda-or-imports-broken" 2
import torch, transformers, scipy, tqdm, sys
print(f"python      = {sys.version.split()[0]}")
print(f"torch       = {torch.__version__}  cuda_build={torch.version.cuda}")
print(f"transformers= {transformers.__version__}  scipy={scipy.__version__}  tqdm={tqdm.__version__}")
assert torch.cuda.is_available(), "FATAL: torch.cuda.is_available() == False"
print(f"GPU         = {torch.cuda.get_device_name(0)}  vram={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# Verify BackCheck path puts the model on CUDA
import os, django
sys.path.insert(0, '/workspace/stego-aas/stego-aas/stegoaas')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stegoaas.settings')
django.setup()
from sparsamp_app.model_manager import get_model_manager
mm = get_model_manager(); mm.load()
dev = str(next(mm.model.parameters()).device)
print(f"BackCheck GPT-2 device = {dev}")
assert dev.startswith('cuda'), f"BackCheck on {dev}, expected cuda"

# Verify STEAD reconstructed-import surface
sys.path.insert(0, '/workspace/stego-aas')
from stead_baseline.wrapper import SteadAdapter
SteadAdapter._verify_upstream_complete()
print("STEAD vendored modules import cleanly")
print("=== SANITY OK ===")
PY
log "sanity OK"

# ----------------------------------------------------------------------------
log "=== Step 3: smoke benchmark N=3 ==="
cd "$REPO"
SMOKE_CSV="$OUT/smoke_N3.csv"
# 64-bit messages (8 chars * 8 bits); STEAD length=1024 leaves plenty of capacity
# headroom (smoke at length=512 saw 71-147 bits cap — 1024 ~doubles that).
python benchmark/run_512bit.py --num-seeds 3 --num-chars 8 --stead-length 1024 \
    --out "$SMOKE_CSV" \
    || mark_done "smoke-crashed" 3

# Decide whether to proceed: any STEAD success in smoke is enough to greenlight.
if ! python -c "
import csv, sys
rows = list(csv.DictReader(open('$SMOKE_CSV')))
stead_rows = [r for r in rows if r['system']=='stead']
stead_succ = sum(1 for r in stead_rows if r['success']=='1')
print(f'smoke STEAD: {stead_succ}/{len(stead_rows)} succeeded')
bc_rows = [r for r in rows if r['system']=='backcheck']
bc_succ = sum(1 for r in bc_rows if r['success']=='1')
print(f'smoke BackCheck: {bc_succ}/{len(bc_rows)} succeeded')
sys.exit(0 if stead_rows and stead_succ >= 1 else 1)
"; then
    log "smoke shows STEAD never succeeded — NOT proceeding to N=50"
    mark_done "smoke-stead-zero-success" 4
fi
log "smoke OK"

# ----------------------------------------------------------------------------
log "=== Step 4: full benchmark N=50 ==="
FULL_CSV="$OUT/full_N50.csv"
python benchmark/run_512bit.py --num-seeds 50 --num-chars 8 --stead-length 1024 \
    --out "$FULL_CSV" \
    || mark_done "full-run-crashed" 5

log "full run finished"
ls -la "$OUT"
mark_done "all-stages-completed" 0
