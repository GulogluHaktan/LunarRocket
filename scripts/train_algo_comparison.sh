#!/usr/bin/env bash
# A2 (see the iniş yol haritası doc): runs SAC/TD3/DDPG training once per
# (algorithm x seed) combination, back-to-back, using ISAACLAB_ALGO to select
# the agent config train_sac.py resolves (see train_isaaclab_docker.sh).
#
# This is a comparison table, not a research contribution on its own -- run
# it last, after B1 (residual action) has settled, so the winning
# configuration's algorithm choice can be added as another column instead of
# re-running everything.
#
# Idempotent / resumable: same JSONL-manifest pattern as
# train_terrain_blend_comparison.sh, one line per (algo, seed) run; a pair
# whose line already has status="ok" is SKIPPED on a re-run.
#
# Sequential, not parallel: all runs want the whole GPU; a single run's
# failure does not abort the others.
#
# Usage:
#   ./scripts/train_algo_comparison.sh
#   ISAACLAB_COMPARISON_SEEDS="42 7 123" ./scripts/train_algo_comparison.sh   # default
#   ISAACLAB_MAX_ITERATIONS=1200 ./scripts/train_algo_comparison.sh   # full-scale instead of the ~350-iter default
#   ISAACLAB_COMPARISON_ALGOS="sac td3" ./scripts/train_algo_comparison.sh   # subset
#
# Requires a working GPU (nvidia-smi succeeding) and the Docker path set up
# per README.md section 2 -- run ./scripts/test_docker_gpu.sh first.
#
# No automated statistical analysis ships with this script (unlike
# train_terrain_blend_comparison.sh's TensorBoard analysis, which is specific
# to the terrain_transition experiment) -- read the per-algo TensorBoard logs
# under each algo's own logs/sb3_<algo>/<task>/ directory.

set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

ALGOS="${ISAACLAB_COMPARISON_ALGOS:-sac td3 ddpg}"
SEEDS="${ISAACLAB_COMPARISON_SEEDS:-42 7 123}"
export ISAACLAB_MAX_ITERATIONS="${ISAACLAB_MAX_ITERATIONS:-350}"
TASK="${ISAACLAB_TASK:-LunarRocket-Lander-Direct-v0}"
RESULTS_DIR="experiments/algo_comparison/results"
MANIFEST="$RESULTS_DIR/comparison_manifest.jsonl"
RUN_LOG="$RESULTS_DIR/run_$(date +%Y-%m-%d_%H-%M-%S).log"

mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "[comparison] started $(date -Iseconds), algos=[$ALGOS], seeds=[$SEEDS], ISAACLAB_MAX_ITERATIONS=$ISAACLAB_MAX_ITERATIONS"

if ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi failed -- GPU driver not usable (see README.md Troubleshooting)." >&2
  echo "Fix the driver (often just a reboot after a driver package update) before running this." >&2
  exit 1
fi
touch "$MANIFEST"

log_root_for_algo() {
  # SAC keeps the historical "sb3_sac" directory name (see train_sac.py);
  # TD3/DDPG get their own algo-named subdirectory.
  echo "logs/sb3_$1/$TASK"
}

already_done() {
  # already_done <algo> <seed> -- true if a status="ok" line exists for this pair
  python3 - "$MANIFEST" "$1" "$2" <<'PYEOF'
import json, sys
manifest, algo, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("algo") == algo and int(row.get("seed", -1)) == seed and row.get("status") == "ok":
                sys.exit(0)
except FileNotFoundError:
    pass
sys.exit(1)
PYEOF
}

failed_runs=()
for seed in $SEEDS; do
  for algo in $ALGOS; do
    if already_done "$algo" "$seed"; then
      echo "[comparison] skip (already done): algo=$algo seed=$seed"
      continue
    fi

    log_root="$(log_root_for_algo "$algo")"
    mkdir -p "$log_root" 2>/dev/null || true  # usually root-owned already, created by the training container

    echo "=================================================================="
    echo "[comparison] starting run: algo=$algo seed=$seed ($(date -Iseconds))"
    echo "=================================================================="
    before=$(ls -1 "$log_root" 2>/dev/null | grep -E '^[0-9]{4}-' || true)
    run_start=$(date +%s)

    ISAACLAB_ALGO="$algo" ISAACLAB_SEED="$seed" ./scripts/train_isaaclab_docker.sh
    run_exit=$?
    if [[ $run_exit -eq 0 ]]; then
      run_status="ok"
    else
      run_status="FAILED (exit $run_exit)"
      failed_runs+=("$algo/seed=$seed")
    fi
    run_elapsed=$(( $(date +%s) - run_start ))
    echo "[comparison] algo=$algo seed=$seed finished: $run_status, elapsed=${run_elapsed}s"

    after=$(ls -1 "$log_root" 2>/dev/null | grep -E '^[0-9]{4}-' || true)
    new_dir=$(comm -13 <(echo "$before" | sort) <(echo "$after" | sort) | tail -1)
    if [[ -z "$new_dir" ]]; then
      echo "WARNING: could not detect a new log directory for algo=$algo seed=$seed (run may not have produced one)" >&2
      printf '{"algo": "%s", "seed": %s, "run_dir": null, "wall_seconds": %d, "status": "no_log_dir"}\n' \
        "$algo" "$seed" "$run_elapsed" >> "$MANIFEST"
      continue
    fi
    echo "[comparison] algo=$algo seed=$seed -> $log_root/$new_dir (${run_elapsed}s wall time)"
    printf '{"algo": "%s", "seed": %s, "run_dir": "%s", "wall_seconds": %d, "status": "%s"}\n' \
      "$algo" "$seed" "$new_dir" "$run_elapsed" "$run_status" >> "$MANIFEST"
  done
done

echo ""
echo "All comparison runs attempted. Manifest: $MANIFEST"
if [[ ${#failed_runs[@]} -gt 0 ]]; then
  echo "WARNING: these (algo/seed) runs reported a failure (see log above for the section): ${failed_runs[*]}"
fi

date -Iseconds > "$RESULTS_DIR/DONE.txt"
echo "manifest=$MANIFEST" >> "$RESULTS_DIR/DONE.txt"
echo "failed_runs=${failed_runs[*]:-none}" >> "$RESULTS_DIR/DONE.txt"
echo "[comparison] DONE. See $RESULTS_DIR/DONE.txt, $RUN_LOG, and the per-algo TensorBoard logs."
