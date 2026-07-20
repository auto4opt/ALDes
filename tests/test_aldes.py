from __future__ import annotations

import numpy as np
import pytest
import torch

import conf
import run_paper_subset
import train as training_script
from EWC import EWC
from autooptlib.aldes import EvaluationConfig, validate_sequence
from autooptlib.aldes.evaluator import (
    _evaluate_pbo_sequences,
    _resolve_evaluation_workers,
)
from models.model.transformer import Transformer
from util import device as device_module


def _model(*, continual: bool = False) -> Transformer:
    return Transformer(
        dec_voc_size=32,
        d_model=32,
        n_head=4,
        max_len=50,
        ffn_hidden=64,
        n_layers=1,
        drop_prob=0.0,
        device=torch.device("cpu"),
        condition_on_features=continual,
    )


def _inputs(batch_size: int = 4):
    target = torch.full((batch_size, 1), 17, dtype=torch.long)
    attention = torch.arange(27).repeat(batch_size, 1)
    return target, attention


def test_device_selection_prioritizes_accelerators(monkeypatch):
    monkeypatch.setattr(device_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(device_module, "_mps_available", lambda: True)
    assert device_module.resolve_device().type == "cuda"

    monkeypatch.setattr(device_module.torch.cuda, "is_available", lambda: False)
    assert device_module.resolve_device().type == "mps"

    monkeypatch.setattr(device_module, "_mps_available", lambda: False)
    assert device_module.resolve_device().type == "cpu"


def test_cli_defaults_to_one_problem_and_one_seed():
    args = training_script.build_parser().parse_args([])
    assert args.mode == "single"
    assert args.problems is None
    assert args.seeds == [1]


def test_cli_rejects_unknown_pbo_problem():
    with pytest.raises(SystemExit):
        training_script.build_parser().parse_args(["--problems", "24"])


def test_paper_reference_data_is_packaged_and_loadable():
    reference_dir = run_paper_subset.DEFAULT_REFERENCE_DIR
    assert reference_dir.is_dir()
    assert run_paper_subset._reference_result(reference_dir, 1).size == 30


def test_main_uses_safe_single_problem_default(monkeypatch):
    calls = []
    training_script.EWC_ = object()
    training_script.current_initial_populations = object()
    monkeypatch.setattr(training_script, "seed_torch", lambda seed: None)
    monkeypatch.setattr(
        training_script,
        "train_separately",
        lambda problems, evaluate_test=False: calls.append((problems, evaluate_test)),
    )

    training_script.main([])

    assert calls == [([1], False)]
    assert training_script.EWC_ is None
    assert training_script.current_initial_populations is None


def test_unavailable_explicit_accelerator_fails_clearly(monkeypatch):
    monkeypatch.setattr(device_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device_module, "_mps_available", lambda: False)
    with pytest.raises(RuntimeError, match="cannot access"):
        device_module.resolve_device("cuda")
    with pytest.raises(RuntimeError, match="does not provide"):
        device_module.resolve_device("mps")


def test_cpu_evaluation_worker_selection(monkeypatch):
    monkeypatch.delenv("ALDES_EVAL_WORKERS", raising=False)
    monkeypatch.setattr("autooptlib.aldes.evaluator.os.cpu_count", lambda: 10)
    assert _resolve_evaluation_workers(None, 16) == 10
    assert _resolve_evaluation_workers(20, 4) == 4
    assert _resolve_evaluation_workers(1, 16) == 1
    monkeypatch.setenv("ALDES_EVAL_WORKERS", "2")
    assert _resolve_evaluation_workers(None, 16) == 2
    with pytest.raises(ValueError, match="positive integer"):
        _resolve_evaluation_workers(0, 4)


def test_parallel_cpu_evaluation_matches_serial():
    actions = np.asarray(
        [
            [17, 0, 29, 8, 29, 12, 29, 18],
            [17, 1, 29, 11, 29, 12, 29, 18],
        ]
    )
    config = EvaluationConfig(
        population_size=4,
        evaluations=20,
        runs=2,
        seed=19,
    )
    serial = _evaluate_pbo_sequences(actions, 1, [1], config, workers=1)
    parallel = _evaluate_pbo_sequences(actions, 1, [1], config, workers=2)

    np.testing.assert_array_equal(serial[0], parallel[0])
    for serial_values, parallel_values in zip(serial[1], parallel[1]):
        np.testing.assert_array_equal(serial_values, parallel_values)


def test_single_problem_mode_is_default_and_ignores_features():
    assert conf.aldes_mode == "single"
    model = _model()
    assert model.decoder.emb.tok_emb.padding_idx is None
    target, attention = _inputs()

    model.eval()
    with torch.no_grad():
        actions, _, sampled_log_probability = model(None, target, attention)
    _, _, replay_log_probability = model(None, target, attention, action=actions)

    for sequence in actions:
        validate_sequence(sequence.tolist())
    torch.testing.assert_close(
        sampled_log_probability, replay_log_probability, rtol=0, atol=1e-6
    )


def test_continual_mode_requires_and_uses_problem_features():
    torch.manual_seed(4)
    model = _model(continual=True)
    target, attention = _inputs()
    zeros = torch.zeros(4, 1, 32)
    ones = torch.ones(4, 1, 32)

    with pytest.raises(ValueError, match="requires problem features"):
        model(None, target, attention)

    model.eval()
    with torch.no_grad():
        actions, _, sampled_log_probability = model(zeros, target, attention)
        _, _, replay_log_probability = model(zeros, target, attention, action=actions)
        _, _, changed_log_probability = model(ones, target, attention, action=actions)

    torch.testing.assert_close(
        sampled_log_probability, replay_log_probability, rtol=0, atol=1e-6
    )
    assert not torch.equal(replay_log_probability, changed_log_probability)


def test_ewc_fisher_and_penalty_are_finite():
    model = _model()
    target, attention = _inputs()
    actions, _, log_probability = model(None, target, attention)
    del actions
    (-log_probability.sum(dim=1).mean()).backward()

    ewc = EWC(model)
    ewc.update_diag_fisher(model)
    assert torch.isfinite(ewc.penalty(model))
    assert all(
        torch.isfinite(value).all() for value in ewc._precision_matrices.values()
    )


class _MemoryLog:
    seed = 1
    problem_id = 1

    def write_log(self, _message):
        pass

    def dump_log(self, _experience):
        pass


def test_one_ppo_update_runs_without_matlab(monkeypatch):
    torch.manual_seed(8)
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    target, attention = _inputs()

    def objective(actions, _problem_id):
        costs = np.asarray(actions.detach().sum(dim=1), dtype=float)
        return costs.tolist(), [np.asarray([cost]) for cost in costs]

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

    assert inferred.shape[0] == 1
    validate_sequence(inferred[0].tolist())
