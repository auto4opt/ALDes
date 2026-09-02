from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_torch(seed: int = 2) -> None:
    """Seed the random-number generators used by ALDes."""

    seed = int(seed)
    if seed < 0 or seed > np.iinfo(np.uint32).max:
        raise ValueError("Seed must be in NumPy's supported range 0..2**32-1.")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    mps = getattr(torch, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        mps.manual_seed(seed)


class RunLogger:
    """Write human-readable logs and raw PPO histories for one process."""

    def __init__(self, root: str | Path = "logs") -> None:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        self.log_dir = Path(root) / timestamp
        self.log_file = self.log_dir / "log.txt"
        self.problem_id = -1
        self.seed = -1
        self.task_id = None

    def write_log(self, info: Any) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        with self.log_file.open("a", encoding="utf-8") as stream:
            stream.write(f"\n{timestamp}: {info}")

    def dump_log(self, experience: Any) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        task_prefix = f"task{self.task_id}_" if self.task_id is not None else ""
        dump_file = self.log_dir / (
            f"{task_prefix}seed{self.seed}_problem{self.problem_id}_training.pkl"
        )
        temporary = dump_file.with_suffix(dump_file.suffix + ".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(experience, stream)
        temporary.replace(dump_file)


__all__ = ["RunLogger", "seed_torch"]
