#!/usr/bin/env bash
# Runs SAC training once per (terrain blend method x seed) combination,
# back-to-back, then automatically analyzes the resulting TensorBoard logs
# with REAL multi-seed statistics (not just a within-run autocorrelated
# window trick -- see experiments/terrain_transition/analyze_training_comparison.py).
#
# Methods: smoothstep / gaussian / hybrid (see experiments/terrain_transition/
# for the CPU-only study this follows up on).
#
# Idempotent / resumable: results are appended to a JSONL manifest, one line
# per (method, seed) run; a (method, seed) pair whose line already has
# status="ok" is SKIPPED on a re-run, so this script can be safely re-invoked
# after an interruption, or extended with new seeds without re-running old
# ones. This is also how it backfills the original single-seed=42 comparison
# (run before multi-seed support existed) instead of discarding it.
#
# Sequential, not parallel: all runs want the whole GPU, and interleaving
# them would confound wall-clock/VRAM comparisons between runs. A single
# run's failure does not abort the others.
#
# Usage:
#   ./scripts/train_terrain_blend_comparison.sh
#   ISAACLAB_COMPARISON_SEEDS="42 7 123" ./scripts/train_terrain_blend_comparison.sh   # default
#   ISAACLAB_MAX_ITERATIONS=1200 ./scripts/train_terrain_blend_comparison.sh   # full-scale instead of the ~350-iter default
#   ISAACLAB_COMPARISON_METHODS="smoothstep gaussian" ./scripts/train_terrain_blend_comparison.sh  # subset
#
# Requires a working GPU (nvidia-smi succeeding) and the Docker path set up
# per README.md section 2 -- run ./scripts/test_docker_gpu.sh first.

set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

METHODS="${ISAACLAB_COMPARISON_METHODS:-smoothstep gaussian hybrid}"
SEEDS="${ISAACLAB_COMPARISON_SEEDS:-42 7 123}"
export ISAACLAB_MAX_ITERATIONS="${ISAACLAB_MAX_ITERATIONS:-350}"
LOG_ROOT="logs/sb3_sac/${ISAACLAB_TASK:-LunarRocket-Lander-Direct-v0}"
RESULTS_DIR="experiments/terrain_transition/results/training_comparison"
# NOTE: $LOG_ROOT is root-owned (the Docker training container writes there
# as root through the bind mount) -- this script runs as the host user and
# cannot create files/symlinks inside it. The manifest therefore lives under
# $RESULTS_DIR (host-user-owned) and records run directory *names* only;
# analyze_training_comparison.py resolves them against $LOG_ROOT itself.
MANIFEST="$RESULTS_DIR/comparison_manifest.jsonl"
OLD_MANIFEST="$RESULTS_DIR/comparison_manifest.json"
RUN_LOG="$RESULTS_DIR/run_$(date +%Y-%m-%d_%H-%M-%S).log"

mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "[comparison] started $(date -Iseconds), methods=[$METHODS], seeds=[$SEEDS], ISAACLAB_MAX_ITERATIONS=$ISAACLAB_MAX_ITERATIONS"
rm -f "$RESULTS_DIR/DONE.txt"

if ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi failed -- GPU driver not usable (see README.md Troubleshooting)." >&2
  echo "Fix the driver (often just a reboot after a driver package update) before running this." >&2
  exit 1
fi

# One-time migration: fold the original single-run manifest (pre-multi-seed
# format: {"method": {"run_dir", "wall_seconds", "status"}}) into the new
# JSONL format as seed=42, so those runs are recognized as already-done and
# never silently re-run/discarded.
if [[ -f "$OLD_MANIFEST" && ! -f "$MANIFEST" ]]; then
  echo "[comparison] migrating legacy $OLD_MANIFEST -> $MANIFEST (seed=42)"
  python3 - "$OLD_MANIFEST" "$MANIFEST" <<'PYEOF'
import json, sys
old, new = sys.argv[1], sys.argv[2]
with open(old) as f:
    data = json.load(f)
with open(new, "w") as f:
    for method, entry in data.items():
        if isinstance(entry, dict):
            row = {"method": method, "seed": 42, **entry}
        else:
            row = {"method": method, "seed": 42, "run_dir": entry, "wall_seconds": None, "status": "ok"}
        f.write(json.dumps(row) + "\n")
PYEOF
fi
touch "$MANIFEST"

already_done() {
  # already_done <method> <seed> -- true if a status="ok" line exists for this pair
  python3 - "$MANIFEST" "$1" "$2" <<'PYEOF'
import json, sys
manifest, method, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("method") == method and int(row.get("seed", -1)) == seed and row.get("status") == "ok":
                sys.exit(0)
except FileNotFoundError:
    pass
sys.exit(1)
PYEOF
}

mkdir -p "$LOG_ROOT" 2>/dev/null || true  # usually already exists (root-owned, created by the training container)
failed_runs=()
for seed in $SEEDS; do
  for method in $METHODS; do
    if already_done "$method" "$seed"; then
      echo "[comparison] skip (already done): method=$method seed=$seed"
      continue
    fi

    echo "=================================================================="
    echo "[comparison] starting run: method=$method seed=$seed ($(date -Iseconds))"
    echo "=================================================================="
    before=$(ls -1 "$LOG_ROOT" 2>/dev/null | grep -E '^[0-9]{4}-' || true)
    run_start=$(date +%s)

    ISAACLAB_TERRAIN_BLEND_METHOD="$method" ISAACLAB_SEED="$seed" ./scripts/train_isaaclab_docker.sh
    run_exit=$?
    if [[ $run_exit -eq 0 ]]; then
      run_status="ok"
    else
      run_status="FAILED (exit $run_exit)"
      failed_runs+=("$method/seed=$seed")
    fi
    run_elapsed=$(( $(date +%s) - run_start ))
    echo "[comparison] method=$method seed=$seed finished: $run_status, elapsed=${run_elapsed}s"

    after=$(ls -1 "$LOG_ROOT" 2>/dev/null | grep -E '^[0-9]{4}-' || true)
    new_dir=$(comm -13 <(echo "$before" | sort) <(echo "$after" | sort) | tail -1)
    if [[ -z "$new_dir" ]]; then
      echo "WARNING: could not detect a new log directory for method=$method seed=$seed (run may not have produced one)" >&2
      printf '{"method": "%s", "seed": %s, "run_dir": null, "wall_seconds": %d, "status": "no_log_dir"}\n' \
        "$method" "$seed" "$run_elapsed" >> "$MANIFEST"
      continue
    fi
    echo "[comparison] method=$method seed=$seed -> $LOG_ROOT/$new_dir (${run_elapsed}s wall time)"
    printf '{"method": "%s", "seed": %s, "run_dir": "%s", "wall_seconds": %d, "status": "%s"}\n' \
      "$method" "$seed" "$new_dir" "$run_elapsed" "$run_status" >> "$MANIFEST"
  done
done

echo ""
echo "All comparison runs attempted. Manifest: $MANIFEST"
if [[ ${#failed_runs[@]} -gt 0 ]]; then
  echo "WARNING: these (method/seed) runs reported a failure (see log above for the section): ${failed_runs[*]}"
fi

echo "[comparison] running analysis..."
VENV_PY="experiments/terrain_transition/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  python3 -m venv experiments/terrain_transition/.venv
  experiments/terrain_transition/.venv/bin/pip install --quiet numpy matplotlib tensorboard scipy
fi
if "$VENV_PY" experiments/terrain_transition/analyze_training_comparison.py; then
  echo "[comparison] analysis OK -> $RESULTS_DIR"
else
  echo "WARNING: analyze_training_comparison.py failed -- manifest/logs are still there for manual analysis." >&2
fi

date -Iseconds > "$RESULTS_DIR/DONE.txt"
echo "manifest=$MANIFEST" >> "$RESULTS_DIR/DONE.txt"
echo "failed_runs=${failed_runs[*]:-none}" >> "$RESULTS_DIR/DONE.txt"
echo "[comparison] DONE. See $RESULTS_DIR/DONE.txt, $RUN_LOG, and $RESULTS_DIR/*.png"
