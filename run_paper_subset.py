"""Run a time-bounded subset of the published ALDes PBO experiment.

Every completed training trial keeps the paper's per-trial protocol intact.
The wall-clock budget only determines whether another whole problem is started.
"""

from __future__ import annotations

import argparse
from importlib.resources import files
import json
import os
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import loadmat
from scipy.stats import mannwhitneyu

import train as aldes_train
from aldes_setting import begin_index
from autooptlib.aldes.vocabulary import TOKEN_BY_INDEX, normalize_sequence
from conf import batch_size_src, clip, device, ppo_epoch, total_epoch
from util.device import describe_device
from util.my_util import seed_torch


DEFAULT_PROBLEMS = (1, 14, 15)
DEFAULT_REFERENCE_DIR = Path(
    str(files("draw.datas.reference_results").joinpath("design", "instance_4"))
)


def _parse_problems(value: str) -> list[int]:
    problems = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not problems or any(problem < 1 or problem > 23 for problem in problems):
        raise argparse.ArgumentTypeError(
            "problems must be comma-separated IDs in 1..23"
        )
    return problems


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _reference_result(reference_dir: Path, problem_id: int) -> np.ndarray:
    source = reference_dir / f"f{problem_id}.mat"
    values = np.asarray(loadmat(source)["res"], dtype=float).reshape(-1)
    # AutoOptLib minimizes the negated IOH objective; the paper reports the
    # original maximization orientation.
    return -values


def _summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "runs": int(values.size),
        "mean": float(np.mean(values)),
        "sample_variance": float(np.var(values, ddof=1)),
        "sample_std": float(np.std(values, ddof=1)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _model_inputs() -> tuple[None, torch.Tensor, torch.Tensor]:
    target = torch.tensor([[begin_index]], device=device).repeat(batch_size_src, 1)
    attention = torch.arange(0, 27, dtype=torch.int, device=device)
    attention = attention.unsqueeze(0).repeat(batch_size_src, 1)
    return None, target, attention


def _run_trial(
    problem_id: int,
    seed: int,
    reference_dir: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    aldes_train.evaluation_round = 0
    aldes_train.logs.seed = seed
    aldes_train.logs.problem_id = problem_id
    seed_torch(seed)

    model, optimizer = aldes_train.get_model("single")
    source, target, attention = _model_inputs()
    action = aldes_train.train(
        model, optimizer, clip, problem_id, source, target, attention
    )

    test_mean, test_values = aldes_train.get_performance(action, problem_id, eval=1)
    del test_mean
    local = -np.asarray(test_values[0], dtype=float).reshape(-1)
    reference = _reference_result(reference_dir, problem_id)
    local_summary = _summary(local)
    reference_summary = _summary(reference)
    mean_gap = float(local_summary["mean"] - reference_summary["mean"])
    relative_gap = mean_gap / max(abs(float(reference_summary["mean"])), 1e-12)
    test = mannwhitneyu(local, reference, alternative="two-sided")

    normalized_action = normalize_sequence(action.detach().cpu().numpy()[0])
    elapsed = time.monotonic() - started
    return {
        "problem_id": problem_id,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "action_tokens": normalized_action,
        "action_names": [TOKEN_BY_INDEX[token].name for token in normalized_action],
        "local_values": local,
        "local": local_summary,
        "paper_reference_values": reference,
        "paper_reference": reference_summary,
        "mean_gap_local_minus_paper": mean_gap,
        "relative_mean_gap": relative_gap,
        "mann_whitney_u": float(test.statistic),
        "mann_whitney_p": float(test.pvalue),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problems",
        type=_parse_problems,
        default=list(DEFAULT_PROBLEMS),
        help="comma-separated PBO IDs (default: 1,14,15)",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--time-budget-minutes", type=float, default=60.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: experiments/paper_subset_<timestamp>",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
    )
    args = parser.parse_args(argv)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("experiments") / f"paper_subset_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    reference_dir = args.reference_dir
    budget_seconds = args.time_budget_minutes * 60.0
    experiment_started = time.monotonic()

    payload: dict[str, Any] = {
        "status": "running",
        "created_at": datetime.now().astimezone().isoformat(),
        "requested_problems": args.problems,
        "seed": args.seed,
        "time_budget_seconds": budget_seconds,
        "protocol": {
            "training_epochs": total_epoch,
            "algorithms_per_epoch": batch_size_src,
            "ppo_updates_per_epoch": ppo_epoch,
            "training_instances": [1, 2, 3],
            "training_runs_per_instance": 5,
            "training_evaluations_per_run": 5_000,
            "test_instance": 4,
            "test_runs": 30,
            "test_evaluations_per_run": 50_000,
            "population_size": 50,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "evaluation_workers": os.environ.get("ALDES_EVAL_WORKERS", "auto"),
            "training_device": describe_device(device),
        },
        "paper_reference_dir": os.path.relpath(reference_dir, Path.cwd()),
        "results": [],
        "skipped_problems": [],
    }
    results_path = output_dir / "results.json"
    _write_json(results_path, payload)

    durations: list[float] = []
    for index, problem_id in enumerate(args.problems):
        elapsed = time.monotonic() - experiment_started
        remaining = budget_seconds - elapsed
        if durations and remaining < statistics.median(durations):
            payload["skipped_problems"] = args.problems[index:]
            break
        result = _run_trial(problem_id, args.seed, reference_dir)
        durations.append(float(result["elapsed_seconds"]))
        payload["results"].append(result)
        payload["elapsed_seconds"] = time.monotonic() - experiment_started
        _write_json(results_path, payload)

    payload["elapsed_seconds"] = time.monotonic() - experiment_started
    payload["status"] = "completed"
    _write_json(results_path, payload)
    print(f"Structured results: {results_path.resolve()}")


if __name__ == "__main__":
    main()
