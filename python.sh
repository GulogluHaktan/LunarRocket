#!/usr/bin/env bash
# Isaac Sim Python wrapper script
# 
# Usage: ./python.sh app/main.py [args...]
#        ./python.sh app/benchmark_terrain_quality.py --config-dir configs
#

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find Isaac Sim Python executable
# Try multiple locations
ISAAC_PYTHON=""

# Check 1: Local venv-isaac-check
if [ -f "$PROJECT_ROOT/.venv-isaac-check/bin/python" ]; then
    ISAAC_PYTHON="$PROJECT_ROOT/.venv-isaac-check/bin/python"
fi

# Check 2: System Isaac Sim (if installed globally)
if [ -z "$ISAAC_PYTHON" ] && command -v isaac-python &> /dev/null; then
    ISAAC_PYTHON="isaac-python"
fi

# Check 3: Try finding in common paths
if [ -z "$ISAAC_PYTHON" ]; then
    for path in ~/.local/share/ov/pkg/isaac-sim-*/python.sh \
                /opt/nvidia/isaac-sim/python.sh \
                ~/isaac-sim/python.sh; do
        if [ -f "$path" ]; then
            # Use the python.sh wrapper from Isaac Sim
            "$path" "$@"
            exit $?
        fi
    done
fi

if [ -z "$ISAAC_PYTHON" ]; then
    echo "ERROR: Isaac Sim Python not found!"
    echo ""
    echo "Install Isaac Sim or use Docker:"
    echo "  ./scripts/isaac_docker_run.sh"
    echo ""
    exit 1
fi

# Execute Python with Isaac Sim environment
exec "$ISAAC_PYTHON" "$@"
