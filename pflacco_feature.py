"""Paper-style landscape features used only by continual ALDes training."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from autooptlib.aldes import PBOFeatureResult
from scipy.spatial.distance import cdist, pdist

ELA_DIR = Path("ela")
# The paper appendix specifies 100 * dimension random-walk samples. Smaller
# factors are useful only for development smoke tests and must be requested
# explicitly so normal continual training cannot silently change protocol.
PAPER_SAMPLE_FACTOR = 100
ELA_SAMPLE_FACTOR = PAPER_SAMPLE_FACTOR
ELA_SCHEMA_VERSION = 1

# Table A3 in the paper appendix. Runtime slots are retained in their published
# positions but fixed to zero below: wall-clock duration is not a landscape
# property and would otherwise make a seeded feature artifact non-reproducible.
PAPER_FEATURE_NAMES = (
    "disp.ratio_mean_02",
    "ela_meta.quad_simple.adj_r2",
    "disp.ratio_mean_05",
    "ela_meta.quad_simple.cond",
    "disp.ratio_mean_10",
    "ela_meta.quad_w_interact.adj_r2",
    "disp.ratio_mean_25",
    "ela_meta.costs_runtime",
    "disp.ratio_median_02",
    "ic.h_max",
    "disp.ratio_median_05",
    "ic.eps_s",
    "disp.ratio_median_10",
    "ic.eps_max",
    "disp.ratio_median_25",
    "ic.eps_ratio",
    "disp.diff_mean_02",
    "ic.m0",
    "disp.diff_mean_05",
    "ic.costs_runtime",
    "ela_meta.lin_simple.adj_r2",
    "nbc.nn_nb.sd_ratio",
    "ela_meta.lin_simple.intercept",
    "nbc.nn_nb.mean_ratio",
    "ela_meta.lin_simple.coef.min",
    "nbc.nn_nb.cor",
    "ela_meta.lin_simple.coef.max",
    "nbc.dist_ratio.coeff_var",
    "ela_meta.lin_simple.coef.max_by_min",
    "nbc.nb_fitness.cor",
    "ela_meta.lin_w_interact.adj_r2",
    "nbc.costs_runtime",
)


def _artifact_metadata(
    problem_id: int,
    *,
    seed: int,
    sample_factor: int,
    kind: str,
) -> dict[str, object]:
    return {
        "schema_version": ELA_SCHEMA_VERSION,
        "problem_id": int(problem_id),
        "seed": int(seed),
        "sample_factor": int(sample_factor),
        "trials": 5,
        "population_size": 50,
        "feature_dimension": 100,
        "training_dimensions": [100, 225, 400],
        "kind": kind,
    }


def _validate_artifact_metadata(
    metadata: object,
    *,
    problem_id: int,
    kind: str,
) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise ValueError("ALDes landscape artifact metadata must be a JSON object.")
    expected = {
        "schema_version": ELA_SCHEMA_VERSION,
        "problem_id": int(problem_id),
        "sample_factor": PAPER_SAMPLE_FACTOR,
        "trials": 5,
        "population_size": 50,
        "feature_dimension": 100,
        "training_dimensions": [100, 225, 400],
        "kind": kind,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(
                f"ALDes {kind} artifact has invalid {name!r}; regenerate it "
                "with the default paper protocol."
            )
    seed = metadata.get("seed")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("ALDes landscape artifact seed must be non-negative.")
    return metadata


def _read_feature_frame(path: Path, problem_id: int) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as stream:
        header = stream.readline()
    prefix = "# ALDES-ELA "
    if not header.startswith(prefix):
        raise ValueError(
            f"{path} has no ALDes protocol metadata; regenerate the artifact."
        )
    try:
        metadata = json.loads(header[len(prefix) :])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} has malformed ALDes protocol metadata.") from exc
    _validate_artifact_metadata(metadata, problem_id=problem_id, kind="features")
    if metadata.get("feature_names") != list(PAPER_FEATURE_NAMES):
        raise ValueError(f"{path} does not declare the paper's factor schema.")
    return pd.read_csv(path, comment="#")


def _binary_random_walk(
    dimension: int, length: int, rng: np.random.Generator
) -> np.ndarray:
    current = rng.integers(0, 2, size=dimension, dtype=np.int8)
    sample = np.empty((length, dimension), dtype=np.int8)
    for row in range(length):
        sample[row] = current
        column = int(rng.integers(0, dimension))
        current[column] = 1 - current[column]
    return sample


def _validate_binary_sample(decisions: np.ndarray, *, min_rows: int = 2) -> np.ndarray:
    sample = np.asarray(decisions)
    if min_rows <= 0:
        raise ValueError("min_rows must be positive.")
    if sample.ndim != 2 or sample.shape[0] < min_rows or sample.shape[1] == 0:
        raise ValueError(f"A binary feature sample needs at least {min_rows} row(s).")
    if not np.isin(sample, (0, 1)).all():
        raise ValueError("PBO feature samples must contain only binary values.")
    return sample


def _hamming_histogram(decisions: np.ndarray, *, block_size: int = 512) -> np.ndarray:
    """Count all pairwise Hamming distances without a quadratic matrix."""

    sample = _validate_binary_sample(decisions, min_rows=1)
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    dimension = sample.shape[1]
    histogram = np.zeros(dimension + 1, dtype=np.int64)
    for left_start in range(0, sample.shape[0], block_size):
        left = sample[left_start : left_start + block_size]
        within = np.rint(pdist(left, metric="hamming") * dimension).astype(int)
        histogram += np.bincount(within, minlength=dimension + 1)
        for right_start in range(left_start + block_size, sample.shape[0], block_size):
            right = sample[right_start : right_start + block_size]
            between = np.rint(cdist(left, right, metric="hamming") * dimension).astype(
                int
            )
            histogram += np.bincount(between.reshape(-1), minlength=dimension + 1)
    return histogram


def _distance_summary(decisions: np.ndarray) -> tuple[float, float]:
    sample = _validate_binary_sample(decisions, min_rows=1)
    if sample.shape[0] < 2:
        return np.nan, np.nan
    counts = _hamming_histogram(sample)
    counts[0] = 0
    total = int(counts.sum())
    if total == 0:
        return np.nan, np.nan
    dimension = sample.shape[1]
    values = np.arange(counts.size)
    mean = float(np.dot(values, counts) / (total * dimension))
    cumulative = np.cumsum(counts)
    lower_rank = (total - 1) // 2 + 1
    upper_rank = total // 2 + 1
    lower = int(np.searchsorted(cumulative, lower_rank))
    upper = int(np.searchsorted(cumulative, upper_rank))
    median = (lower + upper) / (2.0 * dimension)
    return mean, median


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _calculate_information_content_hamming(
    decisions: np.ndarray,
    objectives: np.ndarray,
    *,
    seed: int,
    neighborhood: int = 20,
) -> dict[str, float]:
    """Calculate the selected information-content factors with Hamming distance.

    PFLACCO's public implementation hard-codes Euclidean distance.  That is
    inappropriate for the paper's binary PBO protocol, which explicitly uses
    Hamming distance for factor calculation.  This follows the same greedy
    nearest-neighbour tour and equations while avoiding global RNG mutation.
    """

    sample = _validate_binary_sample(decisions, min_rows=3)
    fitness = np.asarray(objectives, dtype=float).reshape(-1)
    if fitness.shape[0] != sample.shape[0] or not np.isfinite(fitness).all():
        raise ValueError("Feature objectives must be one finite value per sample.")
    if seed < 0 or neighborhood <= 0:
        raise ValueError(
            "Information-content seed must be non-negative and neighborhood positive."
        )

    # PFLACCO averages objective values at duplicate decision vectors before
    # constructing its tour. Preserve first-occurrence order for stable ties.
    locations: dict[bytes, int] = {}
    unique_rows: list[np.ndarray] = []
    objective_sums: list[float] = []
    objective_counts: list[int] = []
    for row, packed, objective in zip(sample, np.packbits(sample, axis=1), fitness):
        key = packed.tobytes()
        index = locations.get(key)
        if index is None:
            locations[key] = len(unique_rows)
            unique_rows.append(row)
            objective_sums.append(float(objective))
            objective_counts.append(1)
        else:
            objective_sums[index] += float(objective)
            objective_counts[index] += 1
    if len(unique_rows) < 3:
        raise ValueError(
            "Information-content factors require at least three unique samples."
        )
    unique = np.asarray(unique_rows, dtype=np.int8)
    unique_fitness = np.divide(objective_sums, objective_counts)

    from sklearn.neighbors import NearestNeighbors

    neighbour_count = min(int(neighborhood), unique.shape[0])
    nearest = NearestNeighbors(
        n_neighbors=neighbour_count,
        algorithm="brute",
        metric="hamming",
    ).fit(unique)
    neighbour_distances, neighbour_indices = nearest.kneighbors(unique)

    start = int(np.random.RandomState(seed).choice(unique.shape[0]))
    permutation = np.empty(unique.shape[0], dtype=int)
    permutation[0] = start
    tour_distances = np.empty(unique.shape[0] - 1, dtype=float)
    remaining = np.ones(unique.shape[0], dtype=bool)
    remaining[start] = False
    current = start
    for position in range(1, unique.shape[0]):
        local_indices = neighbour_indices[current]
        local_positions = np.flatnonzero(remaining[local_indices])
        if local_positions.size:
            local_position = int(local_positions[0])
            next_index = int(local_indices[local_position])
            distance = float(neighbour_distances[current, local_position])
        else:
            candidates = np.flatnonzero(remaining)
            distances = cdist(
                unique[current : current + 1], unique[candidates], metric="hamming"
            ).reshape(-1)
            candidate_position = int(distances.argmin())
            next_index = int(candidates[candidate_position])
            distance = float(distances[candidate_position])
        if distance <= 0:
            raise RuntimeError("Duplicate samples remained in the Hamming tour.")
        permutation[position] = next_index
        tour_distances[position - 1] = distance
        remaining[next_index] = False
        current = next_index

    slopes = np.diff(unique_fitness[permutation]) / tour_distances
    signs = np.sign(slopes)
    magnitudes = np.abs(slopes)
    epsilon = np.insert(10 ** np.linspace(-5, 15, num=1000), 0, 0.0)
    information = np.empty(epsilon.size, dtype=float)
    partial = np.empty(epsilon.size, dtype=float)
    transitions = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0))
    for index, threshold in enumerate(epsilon):
        symbols = np.where(magnitudes < threshold, 0.0, signs)
        left = symbols[:-1]
        right = symbols[1:]
        probabilities = np.asarray(
            [
                np.mean((left == first) & (right == second))
                for first, second in transitions
            ]
        )
        nonzero_probabilities = probabilities[probabilities > 0]
        information[index] = float(
            -np.sum(nonzero_probabilities * np.log(nonzero_probabilities) / np.log(6))
        )
        nonzero_symbols = symbols[symbols != 0]
        changes = (
            int(np.count_nonzero(np.diff(nonzero_symbols)))
            if nonzero_symbols.size
            else 0
        )
        partial[index] = changes / (symbols.size - 1)

    settling = epsilon[information < 0.05]
    eps_s = float(np.log10(settling.min())) if settling.size else np.nan
    maximum_information = float(information.max())
    eps_max = float(np.median(epsilon[information == maximum_information]))
    ratio_candidates = epsilon[partial > 0.5 * partial[0]]
    eps_ratio = (
        float(np.log10(ratio_candidates.max())) if ratio_candidates.size else np.nan
    )
    return {
        "ic.h_max": maximum_information,
        "ic.eps_s": eps_s,
        "ic.eps_max": eps_max,
        "ic.eps_ratio": eps_ratio,
        "ic.m0": float(partial[0]),
        "ic.costs_runtime": 0.0,
    }


def _calculate_nbc_maximize(
    decisions: np.ndarray,
    objectives: np.ndarray,
    *,
    block_size: int = 256,
) -> dict[str, float]:
    """Calculate exact nearest-better features with Hamming distance."""

    sample = _validate_binary_sample(decisions)
    fitness = np.asarray(objectives, dtype=float).reshape(-1)
    if fitness.shape[0] != sample.shape[0] or not np.isfinite(fitness).all():
        raise ValueError("Feature objectives must be one finite value per sample.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    size = sample.shape[0]
    nearest = np.empty(size, dtype=float)
    nearest_better = np.empty(size, dtype=float)
    better_indices = np.full(size, -1, dtype=int)
    all_indices = np.arange(size)
    for start in range(0, size, block_size):
        stop = min(start + block_size, size)
        distances = cdist(sample[start:stop], sample, metric="hamming")
        own = all_indices[start:stop]
        distances[np.arange(stop - start), own] = np.inf
        nearest[start:stop] = distances.min(axis=1)

        eligible = fitness[None, :] > fitness[own, None]
        candidate_distances = np.where(eligible, distances, np.inf)
        has_better = eligible.any(axis=1)
        if bool(has_better.any()):
            rows = np.flatnonzero(has_better)
            indices = candidate_distances[rows].argmin(axis=1)
            global_rows = own[rows]
            better_indices[global_rows] = indices
            nearest_better[global_rows] = candidate_distances[rows, indices]

        no_better_rows = np.flatnonzero(~has_better)
        for row in no_better_rows:
            global_row = own[row]
            equal = (fitness == fitness[global_row]) & (all_indices != global_row)
            if bool(equal.any()):
                tied = np.where(equal, distances[row], np.inf)
                index = int(tied.argmin())
                better_indices[global_row] = index
                nearest_better[global_row] = tied[index]
            else:
                nearest_better[global_row] = nearest[global_row]

    near_std = float(np.std(nearest, ddof=1))
    better_std = float(np.std(nearest_better, ddof=1))
    std_ratio = near_std / better_std if better_std else 0.0
    better_mean = float(np.mean(nearest_better))
    mean_ratio = float(np.mean(nearest)) / better_mean if better_mean else 0.0
    ratios = np.divide(
        nearest,
        nearest_better,
        out=np.zeros_like(nearest),
        where=nearest_better != 0,
    )
    ratio_mean = float(np.mean(ratios))
    ratio_cv = float(np.std(ratios, ddof=1) / ratio_mean) if ratio_mean else 0.0
    indegree = np.bincount(better_indices[better_indices >= 0], minlength=size).astype(
        float
    )
    return {
        "nbc.nn_nb.sd_ratio": std_ratio,
        "nbc.nn_nb.mean_ratio": mean_ratio,
        "nbc.nn_nb.cor": _safe_correlation(nearest, nearest_better),
        "nbc.dist_ratio.coeff_var": ratio_cv,
        # PFLACCO expresses "better" in minimization orientation.  For a PBO
        # maximization objective the transformed fitness is therefore -f.
        "nbc.nb_fitness.cor": _safe_correlation(indegree, -fitness),
        "nbc.costs_runtime": 0.0,
    }


def _calculate_dispersion_maximize(
    decisions: np.ndarray, objectives: np.ndarray
) -> dict[str, float]:
    """Calculate PFLACCO-style dispersion for a maximization problem."""

    sample = _validate_binary_sample(decisions)
    fitness = np.asarray(objectives, dtype=float).reshape(-1)
    if fitness.shape[0] != sample.shape[0] or not np.isfinite(fitness).all():
        raise ValueError("Feature objectives must be one finite value per sample.")
    quantiles = (0.02, 0.05, 0.10, 0.25)
    full_mean, full_median = _distance_summary(sample)
    thresholds = np.quantile(fitness, [1.0 - quantile for quantile in quantiles])
    best_summaries = [
        _distance_summary(sample[fitness >= threshold]) for threshold in thresholds
    ]
    best_means = np.asarray([summary[0] for summary in best_summaries])
    best_medians = np.asarray([summary[1] for summary in best_summaries])

    suffixes = [f"{round(quantile * 100):02d}" for quantile in quantiles]
    groups = (
        ("ratio_mean", best_means / full_mean),
        ("ratio_median", best_medians / full_median),
        ("diff_mean", best_means - full_mean),
        ("diff_median", best_medians - full_median),
    )
    return {
        f"disp.{group}_{suffix}": float(value)
        for group, values in groups
        for suffix, value in zip(suffixes, values)
    }


def _feature_mapping(
    decisions: np.ndarray, objectives: np.ndarray, *, seed: int
) -> dict[str, float]:
    from pflacco.classical_ela_features import calculate_ela_meta

    frame = pd.DataFrame(decisions)
    values = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        values.update(
            _calculate_information_content_hamming(
                decisions,
                objectives,
                seed=seed,
            )
        )
        values.update(calculate_ela_meta(frame, objectives))
        values.update(_calculate_nbc_maximize(decisions, objectives))
        values.update(_calculate_dispersion_maximize(decisions, objectives))
    result = {}
    for name, value in values.items():
        try:
            normalized_name = str(name)
            result[normalized_name] = (
                0.0 if normalized_name.endswith("costs_runtime") else float(value)
            )
        except (TypeError, ValueError):
            continue
    return result


def _extract_pbo_features(
    problem_id: int,
    *,
    dimension: int,
    trials: int,
    sample_factor: int,
    feature_dim: int,
    population_size: int,
    seed: int,
) -> PBOFeatureResult:
    """Extract deterministic maximization-aware PBO landscape features."""

    normalized_problem_id = int(problem_id)
    if normalized_problem_id < 1 or normalized_problem_id > 23:
        raise ValueError("PBO problem IDs must be in 1..23.")
    if seed < 0:
        raise ValueError("Feature seed must be non-negative.")
    if dimension <= 0 or trials <= 0 or sample_factor <= 0:
        raise ValueError("dimension, trials, and sample_factor must be positive.")
    if feature_dim <= 0 or population_size <= 0:
        raise ValueError("feature_dim and population_size must be positive.")
    import ioh

    problem = ioh.get_problem(
        normalized_problem_id,
        instance=1,
        dimension=int(dimension),
        problem_class=ioh.ProblemClass.PBO,
    )
    root = np.random.default_rng(seed)
    mappings = []
    samples = []
    trial_seeds = []
    for _ in range(trials):
        trial_seed = int(root.integers(0, np.iinfo(np.int32).max))
        decisions = _binary_random_walk(
            dimension, sample_factor * dimension, np.random.default_rng(trial_seed)
        )
        objectives = np.asarray(problem(decisions), dtype=float).reshape(-1)
        mappings.append(_feature_mapping(decisions, objectives, seed=trial_seed))
        samples.append(decisions)
        trial_seeds.append(trial_seed)

    names = list(PAPER_FEATURE_NAMES[:feature_dim])
    if feature_dim > len(names):
        names.extend(f"padding.{index}" for index in range(feature_dim - len(names)))
    matrix = np.full((trials, len(names)), np.nan, dtype=float)
    for row, mapping in enumerate(mappings):
        for column, name in enumerate(names):
            matrix[row, column] = mapping.get(name, np.nan)
    finite = np.isfinite(matrix)
    counts = finite.sum(axis=0)
    averaged = np.divide(
        np.where(finite, matrix, 0.0).sum(axis=0),
        counts,
        out=np.zeros(len(names), dtype=float),
        where=counts > 0,
    )
    sample_array = np.stack(samples)
    if sample_array.shape[1] < population_size:
        raise ValueError("The feature sample is smaller than population_size.")
    population_indices = np.stack(
        [
            np.sort(
                np.random.default_rng([trial_seed, 1]).choice(
                    sample_array.shape[1], population_size, replace=False
                )
            )
            for trial_seed in trial_seeds
        ]
    )
    initial = np.stack(
        [sample_array[row, indices] for row, indices in enumerate(population_indices)]
    )
    return PBOFeatureResult(
        np.asarray(averaged[:feature_dim], dtype=np.float32),
        tuple(names[:feature_dim]),
        sample_array,
        initial,
    )


def _sample_walk_populations(
    *,
    dimension: int,
    trials: int,
    sample_factor: int,
    population_size: int,
    seed: int,
) -> np.ndarray:
    """Sample reusable populations from long walks without retaining each walk."""

    length = dimension * sample_factor
    if dimension <= 0 or trials <= 0 or sample_factor <= 0:
        raise ValueError("dimension, trials, and sample_factor must be positive.")
    if population_size <= 0 or population_size > length:
        raise ValueError("population_size must fit inside the sampled walk.")
    root = np.random.default_rng(seed)
    populations = []
    for _ in range(trials):
        trial_seed = int(root.integers(0, np.iinfo(np.int32).max))
        walk_rng = np.random.default_rng(trial_seed)
        selected = np.sort(
            np.random.default_rng([trial_seed, 1]).choice(
                length, population_size, replace=False
            )
        )
        population = np.empty((population_size, dimension), dtype=np.int8)
        current = walk_rng.integers(0, 2, size=dimension, dtype=np.int8)
        selected_position = 0
        for row in range(length):
            if row == selected[selected_position]:
                population[selected_position] = current
                selected_position += 1
                if selected_position == population_size:
                    break
            column = int(walk_rng.integers(0, dimension))
            current[column] = 1 - current[column]
        populations.append(population)
    return np.stack(populations)


def _problem_ids(problem_ids: Iterable[int]) -> list[int]:
    ids = list(dict.fromkeys(int(problem_id) for problem_id in problem_ids))
    if not ids or any(problem_id < 1 or problem_id > 23 for problem_id in ids):
        raise ValueError("PBO problem IDs must be in 1..23.")
    return ids


def _feature_matrix(
    problem_ids: Iterable[int], *, auto_generate: bool
) -> tuple[list[int], np.ndarray]:
    ids = _problem_ids(problem_ids)
    paths = {problem_id: ELA_DIR / f"ela_result{problem_id}.csv" for problem_id in ids}
    missing_ids = [
        problem_id for problem_id, path in paths.items() if not path.exists()
    ]
    if missing_ids:
        if not auto_generate:
            raise FileNotFoundError(
                f"Missing ALDes feature file: {paths[missing_ids[0]]}"
            )
        save_ela(missing_ids)

    frames = {
        problem_id: _read_feature_frame(path, problem_id)
        for problem_id, path in paths.items()
    }
    columns = list(frames[ids[0]].columns)
    if columns != list(PAPER_FEATURE_NAMES):
        raise ValueError(
            "Saved ALDes feature files do not use the paper's 32-factor schema; "
            "regenerate them with pflacco_feature.py."
        )
    for frame in frames.values():
        if len(frame) != 1:
            raise ValueError("Each saved ALDes feature file must contain one row.")
        if list(frame.columns) != columns:
            raise ValueError("Saved ALDes feature files do not share one schema.")
    matrix = np.vstack(
        [frames[problem_id].iloc[0].to_numpy(dtype=float) for problem_id in ids]
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Saved ALDes features must all be finite.")
    return ids, matrix


def cal_feature(solution):
    """Retain the small compatibility helper used by older experiments."""

    from pflacco.classical_ela_features import calculate_ela_meta

    if len(solution) == 0 or len(solution) % 2:
        raise ValueError("solution must contain one or more decision/objective pairs.")
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
    sample_factor: int = ELA_SAMPLE_FACTOR,
) -> None:
    """Extract and persist continual-task features and reusable populations."""

    ids = _problem_ids(problem_ids)
    if seed < 0:
        raise ValueError("Feature seed must be non-negative.")
    if sample_factor <= 0:
        raise ValueError("sample_factor must be positive.")
    ELA_DIR.mkdir(parents=True, exist_ok=True)
    dimensions = {0: 100, 1: 225, 2: 400}
    for problem_id in ids:
        feature_result = _extract_pbo_features(
            int(problem_id),
            dimension=dimensions[0],
            trials=5,
            sample_factor=sample_factor,
            feature_dim=32,
            population_size=50,
            seed=seed + 100 * int(problem_id) + 1,
        )
        feature_path = ELA_DIR / f"ela_result{problem_id}.csv"
        feature_temporary = feature_path.with_suffix(feature_path.suffix + ".tmp")
        feature_metadata = _artifact_metadata(
            problem_id,
            seed=seed,
            sample_factor=sample_factor,
            kind="features",
        )
        feature_metadata["feature_names"] = list(feature_result.feature_names)
        with feature_temporary.open("w", encoding="utf-8") as stream:
            stream.write(
                "# ALDES-ELA "
                + json.dumps(feature_metadata, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
            pd.DataFrame(
                [feature_result.features], columns=feature_result.feature_names
            ).to_csv(stream, index=False)
        feature_temporary.replace(feature_path)
        populations = {0: feature_result.initial_populations}
        for instance in (1, 2):
            populations[instance] = _sample_walk_populations(
                dimension=dimensions[instance],
                trials=5,
                sample_factor=sample_factor,
                population_size=50,
                seed=seed + 100 * int(problem_id) + instance + 1,
            )
        population_path = ELA_DIR / f"initial_population{problem_id}.npz"
        population_temporary = population_path.with_suffix(
            population_path.suffix + ".tmp"
        )
        population_metadata = _artifact_metadata(
            problem_id,
            seed=seed,
            sample_factor=sample_factor,
            kind="populations",
        )
        with population_temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                _metadata=np.asarray(
                    json.dumps(
                        population_metadata,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                ),
                **{
                    f"instance_{instance}": population
                    for instance, population in populations.items()
                },
            )
        population_temporary.replace(population_path)


def transform(*, auto_generate: bool = True) -> np.ndarray:
    """Load and standardize the 23 continual-task feature vectors."""

    _, matrix = _feature_matrix(range(1, 24), auto_generate=auto_generate)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0] = 1.0
    return ((matrix - mean) / scale).astype(np.float32)


def load_standardized_features(
    problem_ids: Iterable[int], *, auto_generate: bool = True
) -> dict[int, np.ndarray]:
    """Fit one fixed feature scaling over a complete continual task."""

    ids, matrix = _feature_matrix(problem_ids, auto_generate=auto_generate)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0] = 1.0
    standardized = ((matrix - mean) / scale).astype(np.float32)
    return {problem_id: standardized[index] for index, problem_id in enumerate(ids)}


def load_initial_populations(problem_id: int) -> dict[int, np.ndarray]:
    """Load the five sampled populations associated with one feature vector."""

    normalized_id = int(problem_id)
    if normalized_id < 1 or normalized_id > 23:
        raise ValueError("PBO problem IDs must be in 1..23.")
    path = ELA_DIR / f"initial_population{normalized_id}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run pflacco_feature.py for continual training first."
        )
    with np.load(path, allow_pickle=False) as payload:
        if "_metadata" not in payload.files:
            raise ValueError(
                f"{path} has no ALDes protocol metadata; regenerate the artifact."
            )
        try:
            metadata = json.loads(str(payload["_metadata"].item()))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{path} has malformed ALDes protocol metadata.") from exc
        _validate_artifact_metadata(
            metadata,
            problem_id=normalized_id,
            kind="populations",
        )
        populations = {
            int(name.rsplit("_", 1)[1]): np.array(payload[name], copy=True)
            for name in payload.files
            if name.startswith("instance_")
        }
    expected_dimensions = {0: 100, 1: 225, 2: 400}
    if set(populations) != set(expected_dimensions):
        raise ValueError("Initial populations must contain training instances 0, 1, 2.")
    for instance, dimension in expected_dimensions.items():
        array = populations[instance]
        if array.ndim != 3 or array.shape[0] != 5 or array.shape[1] != 50:
            raise ValueError(
                "Each initial-population entry must have shape (5, 50, dimension)."
            )
        if array.shape[2] != dimension:
            raise ValueError(
                f"Initial population {instance} must have dimension {dimension}."
            )
        if not np.isin(array, (0, 1)).all():
            raise ValueError("PBO initial populations must contain only binary values.")
    return populations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-factor",
        type=int,
        default=ELA_SAMPLE_FACTOR,
        help="walk length per dimension (paper appendix and default: 100)",
    )
    arguments = parser.parse_args()
    save_ela(sample_factor=arguments.sample_factor)
