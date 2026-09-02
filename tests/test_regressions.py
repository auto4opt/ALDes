from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from autooptlib.aldes import AutoOptEvaluator, EvaluationConfig, validate_sequence
from autooptlib.aldes.evaluator import _evaluate_pbo_sequences
from autooptlib.aldes.problems import make_pbo_problem
from autooptlib.aldes.vocabulary import END_INDEX, allowed_next_tokens
from scipy.spatial.distance import pdist
from torch import nn

import pflacco_feature
import run_paper_subset
import train as training_script
from EWC import EWC
from models.embedding.positional_encoding import PositionalEncoding
from models.layers.multi_head_attention import MultiHeadAttention
from models.model.transformer import Transformer
from util.my_util import RunLogger, seed_torch

LEGAL_ACTION = torch.tensor([[17, 0, 29, 8, 29, 12, 29, 18]])


class _MemoryLog:
    seed = 7
    problem_id = 1

    def write_log(self, _message):
        pass

    def dump_log(self, _experience):
        pass


def _small_model(*, continual: bool = False, dropout: float = 0.0) -> Transformer:
    return Transformer(
        dec_voc_size=32,
        d_model=32,
        n_head=4,
        max_len=50,
        ffn_hidden=64,
        n_layers=1,
        drop_prob=dropout,
        device=torch.device("cpu"),
        condition_on_features=continual,
    )


def _inputs(batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.full((batch_size, 1), 17, dtype=torch.long)
    attention = torch.arange(27).repeat(batch_size, 1)
    return target, attention


def test_test_evaluation_never_reuses_training_populations(monkeypatch):
    captured = []
    populations = object()

    def evaluate(_action, _problem_id, **kwargs):
        captured.append(kwargs)
        return [0.0], [np.zeros(1)]

    monkeypatch.setattr(training_script, "evaluate_pbo_actions", evaluate)
    monkeypatch.setattr(training_script, "logs", _MemoryLog())
    monkeypatch.setattr(training_script, "current_initial_populations", populations)
    monkeypatch.setattr(training_script, "evaluation_round", 0)

    training_script.get_performance(LEGAL_ACTION, 1)
    training_script.get_performance(LEGAL_ACTION, 1, eval=1)

    assert captured[0]["initial_populations"] is populations
    assert captured[1]["initial_populations"] is None
    assert captured[0]["evaluate_test"] is False
    assert captured[1]["evaluate_test"] is True


def test_test_evaluation_does_not_shift_later_training_seeds(monkeypatch):
    captured = []

    def evaluate(_action, _problem_id, **kwargs):
        captured.append(kwargs["seed"])
        return [0.0], [np.zeros(1)]

    log = _MemoryLog()
    log.seed = 2**32 - 1
    monkeypatch.setattr(training_script, "evaluate_pbo_actions", evaluate)
    monkeypatch.setattr(training_script, "logs", log)

    training_script.evaluation_round = 0
    training_script.test_evaluation_round = 0
    training_script.get_performance(LEGAL_ACTION, 1)
    training_script.get_performance(LEGAL_ACTION, 1, eval=1)
    training_script.get_performance(LEGAL_ACTION, 1)
    training_with_diagnostic = [captured[0], captured[2]]
    diagnostic_seed = captured[1]

    captured.clear()
    training_script.evaluation_round = 0
    training_script.test_evaluation_round = 0
    training_script.get_performance(LEGAL_ACTION, 1)
    training_script.get_performance(LEGAL_ACTION, 1)

    assert captured == training_with_diagnostic
    assert diagnostic_seed not in captured
    assert all(0 <= seed <= 2**32 - 1 for seed in captured + [diagnostic_seed])


def test_ewc_uses_empirical_per_sample_fisher_and_half_penalty():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ewc = EWC(model)

    log_likelihoods = torch.stack((2 * model.weight.sum(), 4 * model.weight.sum()))
    ewc.update_diag_fisher(model, log_likelihoods)
    torch.testing.assert_close(
        ewc._precision_matrices["weight"], torch.tensor([[10.0]])
    )

    with torch.no_grad():
        model.weight.fill_(3.0)
    # 1/2 * Fisher(10) * distance(2)^2
    torch.testing.assert_close(ewc.penalty(model), torch.tensor(20.0))


def test_fisher_estimation_does_not_execute_candidate_algorithms(monkeypatch):
    torch.manual_seed(3)
    model = _small_model()
    optimizer = torch.optim.Adam(model.parameters())
    target, attention = _inputs()
    ewc = EWC(model)
    monkeypatch.setattr(training_script, "EWC_", ewc)
    monkeypatch.setattr(
        training_script,
        "get_performance",
        lambda *_args, **_kwargs: pytest.fail("Fisher must not evaluate objectives"),
    )

    training_script.PPO_get_ewc(
        model,
        optimizer,
        1.0,
        1,
        1,
        None,
        0.2,
        None,
        target,
        attention,
        1,
    )

    assert any(value.count_nonzero() for value in ewc._precision_matrices.values())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_final_ppo_epoch_keeps_a_nonzero_learning_rate(monkeypatch):
    torch.manual_seed(5)
    model = _small_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_script.init_lr)
    target, attention = _inputs()

    def objective(actions, _problem_id):
        costs = np.arange(actions.shape[0], dtype=np.float32)
        return costs.tolist(), [np.asarray([cost]) for cost in costs]

    monkeypatch.setattr(training_script, "logs", _MemoryLog())
    monkeypatch.setattr(training_script, "get_performance", objective)
    monkeypatch.setattr(training_script, "EWC_", None)
    training_script.PPO(
        model,
        optimizer,
        clip=1.0,
        total_epoch=2,
        ppo_epoch=1,
        baseline=None,
        clip_coef=0.2,
        src=None,
        trg=target,
        att_src=attention,
        problem_id=1,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(training_script.init_lr / 2)


def test_ppo_gradient_favors_the_lower_cost_algorithm(monkeypatch):
    class ToyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.preference = nn.Parameter(torch.zeros(()))

        def forward(
            self,
            _src,
            _target,
            _attention,
            action=None,
            reference=False,
        ):
            del action, reference
            actions = LEGAL_ACTION.repeat(2, 1)
            log_probabilities = torch.stack(
                (
                    torch.nn.functional.logsigmoid(self.preference),
                    torch.nn.functional.logsigmoid(-self.preference),
                )
            ).reshape(2, 1, 1)
            return actions, log_probabilities.exp(), log_probabilities

    model = ToyPolicy()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    monkeypatch.setattr(training_script, "logs", _MemoryLog())
    monkeypatch.setattr(training_script, "EWC_", None)
    monkeypatch.setattr(
        training_script,
        "get_performance",
        lambda *_args: ([0.0, 2.0], [np.zeros(1), np.zeros(1)]),
    )

    training_script.PPO(
        model,
        optimizer,
        1.0,
        1,
        1,
        None,
        0.2,
        None,
        torch.tensor([[17], [17]]),
        torch.zeros(2, 27, dtype=torch.int32),
        1,
    )

    # Positive preference raises the probability of batch row 0, whose cost
    # is lower than row 1's cost.
    assert model.preference.item() > 0


def test_real_pbo_evaluator_feeds_one_complete_ppo_update(monkeypatch):
    torch.manual_seed(41)
    model = _small_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_script.init_lr)
    target, attention = _inputs(batch_size=2)
    config = EvaluationConfig(
        population_size=4,
        evaluations=8,
        runs=1,
        seed=41,
    )

    def objective(actions, problem_id):
        return _evaluate_pbo_sequences(
            actions.detach().cpu().numpy(),
            problem_id,
            [1],
            config,
            workers=1,
        )

    monkeypatch.setattr(training_script, "logs", _MemoryLog())
    monkeypatch.setattr(training_script, "get_performance", objective)
    monkeypatch.setattr(training_script, "EWC_", None)

    inferred = training_script.PPO(
        model,
        optimizer,
        clip=1.0,
        total_epoch=1,
        ppo_epoch=1,
        baseline=None,
        clip_coef=0.2,
        src=None,
        trg=target,
        att_src=attention,
        problem_id=1,
    )

    validate_sequence(inferred[0].tolist())


def test_ppo_rejects_nonfinite_evaluator_cost(monkeypatch):
    model = _small_model()
    optimizer = torch.optim.Adam(model.parameters())
    target, attention = _inputs()
    monkeypatch.setattr(training_script, "logs", _MemoryLog())
    monkeypatch.setattr(training_script, "EWC_", None)
    monkeypatch.setattr(
        training_script,
        "get_performance",
        lambda *_args: ([0.0, np.inf], [np.zeros(1), np.zeros(1)]),
    )
    with pytest.raises(FloatingPointError, match="non-finite"):
        training_script.PPO(
            model,
            optimizer,
            1.0,
            1,
            1,
            None,
            0.2,
            None,
            target,
            attention,
            1,
        )


def _write_features(
    directory: Path,
    problem_id: int,
    values: np.ndarray,
    *,
    columns: list[str] | tuple[str, ...] = pflacco_feature.PAPER_FEATURE_NAMES,
    sample_factor: int = pflacco_feature.PAPER_SAMPLE_FACTOR,
) -> None:
    metadata = pflacco_feature._artifact_metadata(
        problem_id,
        seed=1,
        sample_factor=sample_factor,
        kind="features",
    )
    metadata["feature_names"] = list(columns)
    path = directory / f"ela_result{problem_id}.csv"
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# ALDES-ELA " + json.dumps(metadata) + "\n")
        pd.DataFrame([values], columns=columns).to_csv(stream, index=False)


def test_feature_scaling_is_fixed_over_the_declared_task(tmp_path, monkeypatch):
    monkeypatch.setattr(pflacco_feature, "ELA_DIR", tmp_path)
    _write_features(tmp_path, 1, np.zeros(32))
    _write_features(tmp_path, 2, np.ones(32))

    features = pflacco_feature.load_standardized_features([1, 2], auto_generate=False)

    np.testing.assert_array_equal(features[1], -np.ones(32, dtype=np.float32))
    np.testing.assert_array_equal(features[2], np.ones(32, dtype=np.float32))


def test_dispersion_uses_best_maximization_values_and_correct_feature_names():
    rng = np.random.default_rng(31)
    decisions = rng.integers(0, 2, size=(100, 8), dtype=np.int8)
    objectives = decisions @ np.arange(1, 9)

    result = pflacco_feature._calculate_dispersion_maximize(decisions, objectives)
    threshold = np.quantile(objectives, 0.75)
    full_mean, full_median = pflacco_feature._distance_summary(decisions)
    best_mean, best_median = pflacco_feature._distance_summary(
        decisions[objectives >= threshold]
    )

    assert result["disp.ratio_mean_25"] == pytest.approx(best_mean / full_mean)
    assert result["disp.ratio_median_25"] == pytest.approx(best_median / full_median)
    assert result["disp.diff_mean_25"] == pytest.approx(best_mean - full_mean)
    assert result["disp.diff_median_25"] == pytest.approx(best_median - full_median)


def test_chunked_hamming_summary_matches_scipy():
    rng = np.random.default_rng(91)
    decisions = rng.integers(0, 2, size=(37, 11), dtype=np.int8)
    distances = pdist(decisions, metric="hamming")
    distances = distances[distances != 0]

    mean, median = pflacco_feature._distance_summary(decisions)

    assert mean == pytest.approx(distances.mean())
    assert median == pytest.approx(np.median(distances))


def test_nearest_better_fitness_correlation_uses_maximization_orientation():
    decisions = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=np.int8,
    )
    objectives = np.array([2, 4, 3, 6, 5, 0, 1, 7], dtype=float)

    result = pflacco_feature._calculate_nbc_maximize(decisions, objectives)

    assert result["nbc.nb_fitness.cor"] == pytest.approx(-0.7356123579206246)


def test_information_content_uses_normalized_hamming_distance():
    decisions = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    objectives = np.array([0, 2, 1, 5, 3, 8, 4, 7], dtype=float)

    original = pflacco_feature._calculate_information_content_hamming(
        decisions, objectives, seed=9
    )
    repeated_dimensions = pflacco_feature._calculate_information_content_hamming(
        np.repeat(decisions, 2, axis=1), objectives, seed=9
    )

    for name in original:
        assert original[name] == pytest.approx(repeated_dimensions[name])


def test_pbo_feature_mapping_marks_nearest_better_as_maximization(monkeypatch):
    import pflacco.classical_ela_features as classical

    calls = []
    monkeypatch.setattr(
        pflacco_feature,
        "_calculate_information_content_hamming",
        lambda *_args, **_kwargs: {"ic.value": 1.0},
    )
    monkeypatch.setattr(
        classical,
        "calculate_ela_meta",
        lambda *_args, **_kwargs: {"ela.value": 2.0},
    )

    def nbc(decisions, objectives):
        calls.append((decisions.copy(), objectives.copy()))
        return {"nbc.value": 3.0}

    monkeypatch.setattr(pflacco_feature, "_calculate_nbc_maximize", nbc)
    decisions = np.arange(80, dtype=np.int8).reshape(10, 8) % 2
    objectives = np.arange(10, dtype=float)
    result = pflacco_feature._feature_mapping(decisions, objectives, seed=2)

    np.testing.assert_array_equal(calls[0][0], decisions)
    np.testing.assert_array_equal(calls[0][1], objectives)
    assert result["nbc.value"] == 3.0


def test_local_pbo_feature_extractor_is_deterministic_and_reusable():
    kwargs = {
        "dimension": 8,
        "trials": 2,
        "sample_factor": 4,
        "feature_dim": 32,
        "population_size": 4,
        "seed": 17,
    }
    first = pflacco_feature._extract_pbo_features(1, **kwargs)
    second = pflacco_feature._extract_pbo_features(1, **kwargs)

    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.samples, second.samples)
    np.testing.assert_array_equal(first.initial_populations, second.initial_populations)
    assert first.features.shape == (32,)
    assert np.isfinite(first.features).all()
    assert first.initial_populations.shape == (2, 4, 8)
    assert first.feature_names == pflacco_feature.PAPER_FEATURE_NAMES
    for index, name in enumerate(first.feature_names):
        if name.endswith("costs_runtime"):
            assert first.features[index] == 0


def test_high_dimension_walk_population_sampling_is_bounded_and_reproducible():
    kwargs = {
        "dimension": 400,
        "trials": 2,
        "sample_factor": 2,
        "population_size": 50,
        "seed": 23,
    }
    first = pflacco_feature._sample_walk_populations(**kwargs)
    second = pflacco_feature._sample_walk_populations(**kwargs)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 50, 400)
    assert np.isin(first, (0, 1)).all()


def test_feature_loader_rejects_nonfinite_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(pflacco_feature, "ELA_DIR", tmp_path)
    values = np.zeros(32)
    values[4] = np.nan
    _write_features(tmp_path, 1, values)
    with pytest.raises(ValueError, match="finite"):
        pflacco_feature.load_standardized_features([1], auto_generate=False)


def test_feature_loader_rejects_stale_or_unknown_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(pflacco_feature, "ELA_DIR", tmp_path)
    columns = [f"old_feature_{index}" for index in range(32)]
    _write_features(tmp_path, 1, np.zeros(32), columns=columns)

    with pytest.raises(ValueError, match="factor schema"):
        pflacco_feature.load_standardized_features([1], auto_generate=False)


def test_feature_loader_rejects_nonpaper_sample_factor(tmp_path, monkeypatch):
    monkeypatch.setattr(pflacco_feature, "ELA_DIR", tmp_path)
    _write_features(tmp_path, 1, np.zeros(32), sample_factor=10)

    with pytest.raises(ValueError, match="sample_factor"):
        pflacco_feature.load_standardized_features([1], auto_generate=False)


def test_initial_population_loader_checks_protocol_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(pflacco_feature, "ELA_DIR", tmp_path)
    metadata = pflacco_feature._artifact_metadata(
        1,
        seed=1,
        sample_factor=pflacco_feature.PAPER_SAMPLE_FACTOR,
        kind="populations",
    )
    np.savez_compressed(
        tmp_path / "initial_population1.npz",
        _metadata=np.asarray(json.dumps(metadata)),
        instance_0=np.zeros((5, 50, 100), dtype=np.int8),
        instance_1=np.zeros((5, 50, 225), dtype=np.int8),
        instance_2=np.zeros((5, 50, 399), dtype=np.int8),
    )
    with pytest.raises(ValueError, match="dimension 400"):
        pflacco_feature.load_initial_populations(1)


def test_saved_paper_feature_artifacts_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(pflacco_feature, "ELA_DIR", tmp_path)

    def extract(problem_id, **kwargs):
        assert problem_id == 1
        assert kwargs["sample_factor"] == pflacco_feature.PAPER_SAMPLE_FACTOR
        return pflacco_feature.PBOFeatureResult(
            features=np.arange(32, dtype=np.float32),
            feature_names=pflacco_feature.PAPER_FEATURE_NAMES,
            samples=np.zeros((5, 100, 100), dtype=np.int8),
            initial_populations=np.zeros((5, 50, 100), dtype=np.int8),
        )

    monkeypatch.setattr(pflacco_feature, "_extract_pbo_features", extract)
    monkeypatch.setattr(
        pflacco_feature,
        "_sample_walk_populations",
        lambda **kwargs: np.zeros(
            (kwargs["trials"], kwargs["population_size"], kwargs["dimension"]),
            dtype=np.int8,
        ),
    )

    pflacco_feature.save_ela([1], seed=3)

    features = pflacco_feature.load_standardized_features([1], auto_generate=False)
    populations = pflacco_feature.load_initial_populations(1)
    np.testing.assert_array_equal(features[1], np.zeros(32, dtype=np.float32))
    assert {key: value.shape for key, value in populations.items()} == {
        0: (5, 50, 100),
        1: (5, 50, 225),
        2: (5, 50, 400),
    }


class _FakeContinualModel:
    def __call__(self, _src, target, _attention, action=None, reference=False):
        del action, reference
        result = LEGAL_ACTION.repeat(target.shape[0], 1)
        probabilities = torch.ones(target.shape[0], result.shape[1] - 1, 1)
        return result, probabilities, probabilities.log()


def _stub_continual_training(monkeypatch, evaluations):
    feature_calls = []

    def features(ids):
        feature_calls.append(list(ids))
        return {
            problem_id: np.full(32, problem_id, dtype=np.float32)
            for problem_id in dict.fromkeys(ids)
        }

    monkeypatch.setattr(training_script, "logs", _MemoryLog())
    monkeypatch.setattr(training_script, "use_ewc", False)
    monkeypatch.setattr(training_script, "load_standardized_features", features)
    monkeypatch.setattr(
        training_script,
        "load_initial_populations",
        lambda problem_id: {"problem": problem_id},
    )
    monkeypatch.setattr(
        training_script,
        "get_model",
        lambda _mode: (_FakeContinualModel(), object()),
    )
    monkeypatch.setattr(
        training_script,
        "train",
        lambda *_args, **_kwargs: LEGAL_ACTION.clone(),
    )

    def evaluate(_action, problem_id, eval=0):
        evaluations.append(
            (problem_id, eval, training_script.current_initial_populations)
        )
        return [0.0], [np.zeros(1)]

    monkeypatch.setattr(training_script, "get_performance", evaluate)
    return feature_calls


def test_continual_training_fits_features_once_and_test_is_opt_in(monkeypatch):
    evaluations = []
    feature_calls = _stub_continual_training(monkeypatch, evaluations)
    training_script.evaluation_round = 99

    training_script.train_in_one([1, 2], evaluate_test=False)

    assert feature_calls == [[1, 2]]
    assert evaluations == []
    assert training_script.current_initial_populations is None
    assert training_script.evaluation_round == 0


def test_continual_test_evaluates_only_seen_tasks_with_fresh_populations(monkeypatch):
    evaluations = []
    _stub_continual_training(monkeypatch, evaluations)

    training_script.train_in_one([1, 2], evaluate_test=True)

    assert [(problem, flag) for problem, flag, _ in evaluations] == [
        (1, 1),
        (1, 1),
        (2, 1),
    ]
    assert all(populations is None for _, _, populations in evaluations)


def test_continual_ewc_replays_each_seen_problem_and_averages_fisher(monkeypatch):
    fisher_calls = []
    ewc_instances = []

    class FakeEWC:
        def __init__(self, _model):
            self._precision_matrices = {"weight": torch.zeros(())}
            ewc_instances.append(self)

    _stub_continual_training(monkeypatch, [])
    monkeypatch.setattr(training_script, "use_ewc", True)
    monkeypatch.setattr(training_script, "EWC", FakeEWC)

    def estimate(*args):
        problem_id = args[-1]
        fisher_calls.append(problem_id)
        training_script.EWC_._precision_matrices["weight"] += 1

    monkeypatch.setattr(training_script, "PPO_get_ewc", estimate)

    training_script.train_in_one([1, 2])

    assert fisher_calls == [1, 1, 2]
    assert len(ewc_instances) == 2
    assert ewc_instances[0]._precision_matrices["weight"].item() == 1
    assert ewc_instances[1]._precision_matrices["weight"].item() == 1


def test_default_continual_problem_sets_are_independent_tasks(monkeypatch):
    calls = []
    monkeypatch.setattr(training_script, "logs", _MemoryLog())
    monkeypatch.setattr(training_script, "seed_torch", lambda _seed: None)
    monkeypatch.setattr(
        training_script,
        "train_in_one",
        lambda sequence, **kwargs: calls.append((sequence, kwargs)),
    )

    training_script.main(["--mode", "continual"])

    assert [call[0] for call in calls] == [
        list(training_script.continual_problem_sets[0]),
        list(training_script.continual_problem_sets[1]),
    ]


def test_multi_seed_checkpoint_directories_do_not_overwrite(monkeypatch, tmp_path):
    paths = []
    monkeypatch.setattr(training_script, "logs", _MemoryLog())
    monkeypatch.setattr(training_script, "seed_torch", lambda _seed: None)
    monkeypatch.setattr(
        training_script,
        "train_in_one",
        lambda _sequence, **kwargs: paths.append(kwargs["checkpoint_dir"]),
    )

    training_script.main(
        [
            "--mode",
            "continual",
            "--problems",
            "1,2",
            "--seeds",
            "1,2",
            "--checkpoint-dir",
            str(tmp_path),
        ]
    )

    assert paths == [tmp_path / "seed1", tmp_path / "seed2"]


def test_default_continual_tasks_do_not_overwrite_training_histories(tmp_path):
    logger = RunLogger(tmp_path)
    logger.seed = 1
    logger.problem_id = 1
    logger.task_id = 1
    logger.dump_log(["first task"])
    logger.task_id = 2
    logger.dump_log(["second task"])

    assert (logger.log_dir / "task1_seed1_problem1_training.pkl").is_file()
    assert (logger.log_dir / "task2_seed1_problem1_training.pkl").is_file()


def test_single_problem_trials_reset_seed_for_each_problem(monkeypatch):
    seeds = []
    monkeypatch.setattr(training_script, "logs", _MemoryLog())
    monkeypatch.setattr(training_script, "seed_torch", seeds.append)
    monkeypatch.setattr(
        training_script,
        "get_model",
        lambda _mode: (object(), object()),
    )
    monkeypatch.setattr(
        training_script,
        "train",
        lambda *_args, **_kwargs: LEGAL_ACTION.clone(),
    )

    training_script.train_separately([1, 2])

    assert seeds == [7, 7]


def test_continual_model_prepends_feature_token_and_uses_causal_mask(monkeypatch):
    model = _small_model(continual=True)
    target, attention = _inputs()
    features = torch.zeros(2, 1, 32)
    observed = []
    original = model.decoder.layers[0].forward

    def capture(decoder_input, trg_mask):
        observed.append((decoder_input.shape[1], trg_mask.detach().clone()))
        return original(decoder_input, trg_mask)

    monkeypatch.setattr(model.decoder.layers[0], "forward", capture)
    model.eval()
    with torch.no_grad():
        actions, _, _ = model(features, target, attention, reference=True)

    validate_sequence(actions[0].tolist())
    first_length, first_mask = observed[0]
    assert first_length == 2  # feature token followed by begin token
    assert torch.equal(first_mask, torch.tril(torch.ones_like(first_mask)))


def test_model_validates_configuration_features_and_replay():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(31, 4)

    encoding = PositionalEncoding(31, 8, torch.device("cpu"))
    assert encoding(torch.zeros(1, 2, dtype=torch.long)).shape == (2, 31)

    model = _small_model()
    target, attention = _inputs()
    with pytest.raises(ValueError, match="does not accept features"):
        model(torch.zeros(2, 1, 32), target, attention)

    invalid = LEGAL_ACTION.repeat(2, 1)
    invalid[:, 1] = 19
    with pytest.raises(ValueError, match="grammar-invalid"):
        model(None, target, attention, action=invalid)

    fractional = LEGAL_ACTION.repeat(2, 1).to(torch.float32)
    fractional[:, 1] = 0.5
    with pytest.raises(ValueError, match="integer token"):
        model(None, target, attention, action=fractional)

    with pytest.raises(ValueError, match="ffn_hidden"):
        Transformer(
            dec_voc_size=32,
            d_model=32,
            n_head=4,
            max_len=50,
            ffn_hidden=0,
            n_layers=1,
            drop_prob=0.0,
            device=torch.device("cpu"),
        )


def test_replay_resumes_after_a_legal_multitoken_prefix():
    model = _small_model()
    model.eval()
    prefix = LEGAL_ACTION[:, :3].repeat(2, 1)
    replay = LEGAL_ACTION.repeat(2, 1)
    attention = torch.arange(27).repeat(2, 1)

    completed, probabilities, log_probabilities = model(
        None, prefix, attention, action=replay
    )

    torch.testing.assert_close(completed, replay)
    assert probabilities.shape[1] == replay.shape[1] - prefix.shape[1]
    assert log_probabilities.shape == probabilities.shape

    with pytest.raises(ValueError, match="must not already contain the end token"):
        model(None, replay, attention)


def test_random_grammar_walks_always_terminate_and_validate():
    rng = np.random.default_rng(20260902)
    for _ in range(2_000):
        sequence = [17]
        for _ in range(49):
            allowed = np.flatnonzero(allowed_next_tokens(sequence))
            assert allowed.size
            if sequence[-1] == END_INDEX:
                break
            sequence.append(int(rng.choice(allowed)))
        assert sequence[-1] == END_INDEX
        validate_sequence(sequence)


def _component_covering_sequences() -> list[list[int]]:
    sequences = [[17, choose, 29, 8, 29, 12, 29, 18] for choose in range(4)]
    for search in range(4, 12):
        sequence = [17, 0, 29, search]
        if search in {6, 7, 9, 10}:
            sequence.append(19)
        sequence.append(29)
        if search in {4, 5, 6, 7}:
            sequence.extend((8, 29))
        sequence.extend((12, 29, 18))
        sequences.append(sequence)
    for update in range(12, 17):
        sequence = [17, 0, 29, 8, 29, update]
        if update == 16:
            sequence.append(19)
        sequence.extend((29, 18))
        sequences.append(sequence)
    sequences.extend(
        (
            [17, 0, 29, 8, 30, 19, 12, 29, 18],
            [17, 0, 31, 21, 8, 29, 12, 29, 18],
            [17, 0, 31, 22, 4, 29, 8, 29, 12, 29, 18],
        )
    )
    return sequences


def test_every_component_and_pointer_mode_executes():
    sequences = _component_covering_sequences()
    for sequence in sequences:
        validate_sequence(sequence)
    evaluator = AutoOptEvaluator(
        make_pbo_problem(1),
        [1],
        config=EvaluationConfig(population_size=4, evaluations=8, runs=1, seed=11),
    )
    means, performances = evaluator.evaluate_many(sequences)
    assert means.shape == (len(sequences),)
    assert np.isfinite(means).all()
    assert all(values.shape == (1, 1) for values in performances)


@pytest.mark.parametrize("problem_id", range(1, 24))
def test_all_pbo_problems_execute_with_a_tiny_budget(problem_id):
    evaluator = AutoOptEvaluator(
        make_pbo_problem(problem_id),
        [1],
        config=EvaluationConfig(
            population_size=4, evaluations=8, runs=1, seed=problem_id
        ),
    )
    result = evaluator.evaluate(LEGAL_ACTION[0].numpy())
    assert result.shape == (1, 1)
    assert np.isfinite(result).all()


def test_paper_runner_rejects_invalid_seed_and_budget():
    with pytest.raises(SystemExit):
        run_paper_subset.main(["--seed", "-1"])
    with pytest.raises(SystemExit):
        run_paper_subset.main(["--time-budget-minutes", "nan"])


def test_zero_seed_is_supported_consistently():
    assert training_script._seed_list("0,4294967295,0") == [0, 2**32 - 1]
    assert run_paper_subset._seed("0") == 0
    seed_torch(0)

    with pytest.raises(ValueError, match=r"0..2\*\*32-1"):
        seed_torch(-1)


def test_paper_runner_records_failed_status(tmp_path, monkeypatch):
    output = tmp_path / "failed-run"
    monkeypatch.setattr(
        run_paper_subset,
        "_run_trial",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )

    with pytest.raises(RuntimeError, match="synthetic failure"):
        run_paper_subset.main(
            [
                "--problems",
                "1",
                "--output-dir",
                str(output),
                "--time-budget-minutes",
                "1",
            ]
        )

    payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error"] == "RuntimeError: synthetic failure"
    assert payload["environment"]["autooptlib"]


def test_paper_runner_records_completed_status_incrementally(tmp_path, monkeypatch):
    output = tmp_path / "completed-run"
    monkeypatch.setattr(
        run_paper_subset,
        "_run_trial",
        lambda problem_id, seed, _reference_dir: {
            "problem_id": problem_id,
            "seed": seed,
            "elapsed_seconds": 0.01,
        },
    )

    run_paper_subset.main(
        [
            "--problems",
            "1",
            "--output-dir",
            str(output),
            "--time-budget-minutes",
            "1",
        ]
    )

    payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["results"][0]["problem_id"] == 1
    assert payload["elapsed_seconds"] >= 0


def test_missing_reference_is_rejected_before_output_creation(tmp_path):
    output = tmp_path / "must-not-exist"
    with pytest.raises(FileNotFoundError, match="reference"):
        run_paper_subset.main(
            [
                "--problems",
                "1",
                "--reference-dir",
                str(tmp_path / "missing"),
                "--output-dir",
                str(output),
            ]
        )
    assert not output.exists()


def test_plot_log_parsers_preserve_signs_and_scientific_notation(tmp_path):
    pytest.importorskip("matplotlib")
    from draw.util import read_data, read_data_for_transformer

    policy_log = tmp_path / "policy.log"
    policy_log.write_text(
        "total env-steps 1e3\nreturn mean -1.25e+02 variance +2.5e-1\n",
        encoding="utf-8",
    )
    assert read_data(policy_log) == ([1000], [-125.0], [0.25])

    transformer_log = tmp_path / "transformer.log"
    transformer_log.write_text(
        "step : 10 % , loss : -1.2e+05 , cost_mean : +3.5E-2 Training\n",
        encoding="utf-8",
    )
    assert read_data_for_transformer(transformer_log) == [[10.0, -120000.0, 0.035]]


def test_plot_notebooks_compile_from_the_repository_root():
    notebooks = sorted(Path("draw").glob("*.ipynb"))
    assert len(notebooks) == 6
    for notebook in notebooks:
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        sources = []
        for index, cell in enumerate(payload.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            compile(source, f"{notebook}:cell-{index}", "exec")
            sources.append(source)
        combined = "\n".join(sources)
        assert "from util import" not in combined
        assert "datas/logs" not in combined
        assert "performance\\" not in combined

    for source in (
        "draw/datas/txts/train_in_one_seed1_5.txt",
        "draw/datas/txts/spearately_for_train_in_one_problem_seed1_5.txt",
        "draw/datas/txts/spearately_for_not_in_train_in_one.txt",
        "draw/datas/pkls/problem21_04_24_18_57.pkl",
        "draw/datas/pkls/problem21_04_24_20_39.pkl",
        "draw/datas/pkls/problem21_04_24_22_14.pkl",
        "draw/datas/mats/ArchSolution/f1.mat",
        "draw/datas/mats/ArchSolution/f13.mat",
        "draw/datas/mats/ArchSolution/f15.mat",
        "draw/datas/mats/ArchSolution/f20.mat",
    ):
        assert Path(source).is_file(), source
