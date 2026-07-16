from __future__ import annotations

import os
import runpy
from pathlib import Path

import lunar_rocket_lab.tasks  # noqa: F401


def main() -> None:
    isaaclab_root = Path(os.environ.get("ISAACLAB_ROOT", "/workspace/IsaacLab")).resolve()
    train_script = isaaclab_root / "scripts/reinforcement_learning/sb3/train.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Isaac Lab SB3 train script not found: {train_script}")
    runpy.run_path(str(train_script), run_name="__main__")


if __name__ == "__main__":
    main()

