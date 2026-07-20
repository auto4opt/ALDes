"""Paper-style landscape features used only by continual ALDes training."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from autooptlib.aldes import extract_pbo_features

ELA_DIR = Path("ela")


def cal_feature(solution):
    """Retain the small compatibility helper used by older experiments."""

    from pflacco.classical_ela_features import calculate_ela_meta

    result = None
    for index in range(0, len(solution), 2):
        decisions = pd.DataFrame(np.asarray(solution[index])[:10, :])
        objectives = np.asarray(solution[index + 1])[:10].squeeze()
        result = calculate_ela_meta(decisions, objectives)
    return result


def save_ela(
    problem_ids: Iterable[int] = range(1, 24),
    *,
    seed: int = 1,
) -> None:
    """Extract and persist continual-task features and reusable populations."""

    ELA_DIR.mkdir(parents=True, exist_ok=True)
    dimensions = {1: 100, 2: 225, 3: 400}
    for problem_id in problem_ids:
        results = {
            instance: extract_pbo_features(
                int(problem_id),
                instance=1,
                dimension=dimension,
                trials=5,
                sample_factor=100,
                feature_dim=32,
                population_size=50,
                seed=seed + 100 * int(problem_id) + instance,
            )
            for instance, dimension in dimensions.items()
        }
        result = results[1]
        if any(item.feature_names != result.feature_names for item in results.values()):
            raise ValueError("PBO dimensions produced incompatible feature schemas.")
        averaged_features = np.mean(
            np.vstack([item.features for item in results.values()]), axis=0
        )
        pd.DataFrame([averaged_features], columns=result.feature_names).to_csv(
            ELA_DIR / f"ela_result{problem_id}.csv", index=False
        )
        np.savez_compressed(
            ELA_DIR / f"initial_population{problem_id}.npz",
            **{
                f"instance_{instance - 1}": item.initial_populations
                for instance, item in results.items()
            },
        )


def transform(*, auto_generate: bool = True) -> np.ndarray:
    """Load and standardize the 23 continual-task feature vectors."""

    paths = [ELA_DIR / f"ela_result{problem}.csv" for problem in range(1, 24)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        if not auto_generate:
            raise FileNotFoundError(f"Missing ALDes feature file: {missing[0]}")
        save_ela()
    frames = [pd.read_csv(path) for path in paths]
    columns = list(frames[0].columns)
    if any(list(frame.columns) != columns for frame in frames[1:]):
        raise ValueError("Saved ALDes feature files do not share one schema.")
    matrix = np.vstack([frame.iloc[0].to_numpy(dtype=float) for frame in frames])
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0] = 1.0
    return ((matrix - mean) / scale).astype(np.float32)


def load_standardized_features(
    problem_ids: Iterable[int], *, auto_generate: bool = True
) -> dict[int, np.ndarray]:
    """Fit feature scaling on only the continual tasks seen so far."""

    ids = list(dict.fromkeys(int(problem_id) for problem_id in problem_ids))
    if not ids:
        raise ValueError("At least one seen problem is required.")
    paths = {problem_id: ELA_DIR / f"ela_result{problem_id}.csv" for problem_id in ids}
    missing_ids = [
        problem_id for problem_id, path in paths.items() if not path.exists()
    ]
    if missing_ids:
        if not auto_generate:
            raise FileNotFoundError(f"Missing {paths[missing_ids[0]]}")
        save_ela(missing_ids)
    frames = {problem_id: pd.read_csv(path) for problem_id, path in paths.items()}
    columns = list(frames[ids[0]].columns)
    if any(list(frame.columns) != columns for frame in frames.values()):
        raise ValueError("Saved ALDes feature files do not share one schema.")
    matrix = np.vstack(
        [frames[problem_id].iloc[0].to_numpy(dtype=float) for problem_id in ids]
    )
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0] = 1.0
    standardized = ((matrix - mean) / scale).astype(np.float32)
    return {problem_id: standardized[index] for index, problem_id in enumerate(ids)}


def load_initial_populations(problem_id: int) -> dict[int, np.ndarray]:
    """Load the five sampled populations associated with one feature vector."""

    path = ELA_DIR / f"initial_population{int(problem_id)}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run pflacco_feature.py for continual training first."
        )
    with np.load(path, allow_pickle=False) as payload:
        return {
            int(name.rsplit("_", 1)[1]): np.array(payload[name], copy=True)
            for name in payload.files
        }


if __name__ == "__main__":
    save_ela()
