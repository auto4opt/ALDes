import argparse
import time
from pathlib import Path

import numpy as np
import torch
from autooptlib.aldes import evaluate_pbo_actions
from torch import nn
from torch.optim import Adam

from aldes_setting import begin_index, total_index
from conf import (
    adam_eps,
    aldes_mode,
    batch_size_src,
    clip,
    continual_problem_sets,
    d_model,
    device,
    drop_prob,
    ewc_weight,
    ffn_hidden,
    init_lr,
    max_len,
    n_heads,
    n_layers,
    ppo_epoch,
    total_epoch,
    use_ewc,
    weight_decay,
)
from EWC import EWC
from models.model.transformer import Transformer
from pflacco_feature import (
    load_initial_populations,
    load_standardized_features,
)
from util.device import describe_device
from util.my_util import RunLogger, seed_torch

EWC_ = None
current_initial_populations = None
evaluation_round = 0
test_evaluation_round = 0

logs = RunLogger()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, "weight") and m.weight is not None and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)


def get_model(mode=None):
    mode = aldes_mode if mode is None else str(mode).lower()
    if mode not in {"single", "continual"}:
        raise ValueError("ALDes mode must be 'single' or 'continual'.")
    model = Transformer(
        d_model=d_model,
        dec_voc_size=total_index,
        max_len=max_len,
        ffn_hidden=ffn_hidden,
        n_head=n_heads,
        n_layers=n_layers,
        drop_prob=drop_prob,
        device=device,
        condition_on_features=(mode == "continual"),
    ).to(device)

    print(
        f"The model has {count_parameters(model):,} trainable parameters "
        f"on {describe_device(device)}"
    )
    model.apply(initialize_weights)
    optimizer = Adam(
        params=model.parameters(),
        lr=init_lr,
        weight_decay=weight_decay,
        eps=adam_eps,
    )
    return model, optimizer


def get_performance(action, problem_id, eval=0):
    """Evaluate generated actions with the Python AutoOptLib backend."""
    global evaluation_round, test_evaluation_round
    evaluate_test = bool(eval)
    evaluation_seed = None
    if logs.seed >= 0:
        # Training and diagnostic test evaluations need independent random
        # streams. Otherwise enabling the continual forgetting-matrix test
        # shifts every later training seed and changes the policy being
        # measured. SeedSequence also keeps the derived value in uint32 range
        # when the user supplies the largest supported base seed.
        counter = test_evaluation_round if evaluate_test else evaluation_round
        evaluation_seed = int(
            np.random.SeedSequence(
                [int(logs.seed), int(evaluate_test), counter]
            ).generate_state(1, dtype=np.uint32)[0]
        )
        if evaluate_test:
            test_evaluation_round += 1
        else:
            evaluation_round += 1
    mean_performances, performances = evaluate_pbo_actions(
        action,
        problem_id,
        evaluate_test=evaluate_test,
        seed=evaluation_seed,
        # Landscape samples are reusable initial populations only for the
        # 100/225/400-dimensional training instances. Test instance 4 is
        # 625-dimensional and must receive a fresh population.
        initial_populations=(None if bool(eval) else current_initial_populations),
        workers=None,
    )
    if evaluate_test:
        for mean_performance, performance in zip(mean_performances, performances):
            logs.write_log(
                "problem_" + str(problem_id) + " result:\n" + str(performance)
            )
            logs.write_log(
                "problem_" + str(problem_id) + " result mean:\n" + str(mean_performance)
            )

    return mean_performances, performances


def PPO(
    model,
    optimizer,
    clip,
    total_epoch,
    ppo_epoch,
    baseline,
    clip_coef,
    src,
    trg,
    att_src,
    problem_id,
):
    if total_epoch <= 0 or ppo_epoch <= 0:
        raise ValueError("total_epoch and ppo_epoch must be positive.")
    if clip <= 0:
        raise ValueError("Gradient clip must be positive.")
    if not 0 < clip_coef < 1:
        raise ValueError("PPO clip coefficient must be between zero and one.")
    ewc_loss = None
    log_performances = []
    # ppo
    for i in range(total_epoch):
        # Keep the final update non-zero; i / (epochs - 1) wasted the entire
        # final paper-protocol epoch at a learning rate of exactly zero.
        progress = i / max(1, total_epoch)
        learning_rate = init_lr * (1.0 - progress)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        since = time.time()
        model.eval()
        with torch.no_grad():
            action, action_p, action_log_p = model(src, trg, att_src)
            action_log_p = torch.squeeze(action_log_p, 2)
            logs.write_log("action in train: \n" + str(action[0:5]))
            logs.write_log(
                "action_p in train: \n " + str(torch.squeeze(action_p[0:5], 2))
            )

        mean_performances, performances = get_performance(action, problem_id)
        log_performances.append(performances)
        logs.write_log("performances in train: \n " + str(performances[0:5]))
        model_device = next(model.parameters()).device
        cost = torch.as_tensor(
            mean_performances, dtype=action_log_p.dtype, device=model_device
        )
        if cost.ndim != 1 or cost.shape[0] != action.shape[0]:
            raise ValueError("Evaluator returned one cost per generated algorithm.")
        if not torch.isfinite(cost).all():
            raise FloatingPointError("Evaluator returned a non-finite training cost.")
        if baseline is None:
            baseline = cost.mean()
        else:
            baseline = 0.8 * baseline + 0.2 * cost.mean()
        baseline = baseline.detach()

        # ppo update
        for _ in range(ppo_epoch):
            _, _, new_action_log_p = model(src, trg, att_src, action)
            new_action_log_p = torch.squeeze(new_action_log_p, 2)
            logratio = new_action_log_p.sum(1) - action_log_p.sum(1)
            ratio = logratio.exp()

            pg_loss1 = (cost - baseline) * ratio
            pg_loss2 = (cost - baseline) * torch.clamp(
                ratio, 1 - clip_coef, 1 + clip_coef
            )
            loss = torch.max(pg_loss1, pg_loss2).mean()

            if EWC_ is not None:
                ewc_loss = EWC_.penalty(model)
                loss += ewc_weight * ewc_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("PPO produced a non-finite loss.")
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip, error_if_nonfinite=True
            )
            optimizer.step()

        if ewc_loss is not None:
            logs.write_log(
                (
                    "step :",
                    round((i / total_epoch) * 100, 2),
                    "% , ewc_loss :",
                    ewc_loss.item(),
                ).__str__()
            )
            print(
                "step :",
                round((i / total_epoch) * 100, 2),
                "% , ewc_loss :",
                ewc_loss.item(),
            )

        time_elapsed = time.time() - since
        elapsed_message = (
            f",Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )

        print(
            "step :",
            round((i / total_epoch) * 100, 2),
            "% , loss :",
            loss.item(),
            ", cost_mean :",
            cost.mean().item(),
            ", baseline :",
            baseline.item(),
            elapsed_message,
        )
        logs.write_log(
            (
                "step :",
                round((i / total_epoch) * 100, 2),
                "% , loss :",
                loss.item(),
                ", cost_mean :",
                cost.mean().item(),
                ", baseline :",
                baseline.item(),
                elapsed_message,
            ).__str__()
        )

    logs.dump_log(log_performances)
    model.eval()
    with torch.no_grad():
        inferred_action, _, _ = model(src, trg, att_src, reference=True)
    return inferred_action[:1]


def PPO_get_ewc(
    model,
    optimizer,
    clip,
    total_epoch,
    ppo_epoch,
    baseline,
    clip_coef,
    src,
    trg,
    att_src,
    problem_id,
):
    del optimizer, clip, total_epoch, ppo_epoch, baseline, clip_coef, problem_id
    if EWC_ is None:
        raise RuntimeError("EWC must be initialized before estimating Fisher.")
    model.eval()
    with torch.no_grad():
        action, _, _ = model(src, trg, att_src)
    _, _, new_action_log_p = model(src, trg, att_src, action)
    sequence_log_likelihoods = torch.squeeze(new_action_log_p, 2).sum(dim=1)
    EWC_.update_diag_fisher(model, sequence_log_likelihoods)
    return action


def train(model, optimizer, clip, problem_id, src, trg, att_src):
    # PPO old/new policy likelihoods must use the same deterministic network
    # mode; gradients still propagate while the module is in eval mode.
    model.eval()
    baseline = None
    clip_coef = 0.2
    return PPO(
        model,
        optimizer,
        clip,
        total_epoch,
        ppo_epoch,
        baseline,
        clip_coef,
        src,
        trg,
        att_src,
        problem_id,
    )


def train_separately(problem_ids=None, *, evaluate_test=False):
    """Train independent policies for one or more PBO problems.

    The default is intentionally one problem. Single-problem design returns
    the inferred action directly and does not need a model checkpoint.
    """

    global EWC_, current_initial_populations, evaluation_round, test_evaluation_round
    EWC_ = None
    current_initial_populations = None
    print("Train separately")
    logs.write_log("Train separately")
    problem_set = (
        [1]
        if problem_ids is None
        else list(dict.fromkeys(int(item) for item in problem_ids))
    )
    if not problem_set or any(
        problem_id < 1 or problem_id > 23 for problem_id in problem_set
    ):
        raise ValueError(
            "Independent PBO problem IDs must be a non-empty list in 1..23."
        )
    actions = {}
    for problem_id in problem_set:
        # A (problem, seed) trial must be independent of which other problems
        # happened to precede it in the same CLI invocation.
        if logs.seed >= 0:
            seed_torch(logs.seed)
        evaluation_round = 0
        test_evaluation_round = 0
        logs.problem_id = problem_id
        print("Train problem_" + problem_id.__str__())
        logs.write_log("Train problem_" + problem_id.__str__())

        model, optimizer = get_model("single")

        src = None
        trg = torch.tensor([begin_index]).to(device)
        trg = trg.unsqueeze(dim=0)
        # train with batch
        trg = trg.repeat(batch_size_src, 1)
        att_src = torch.arange(0, 27, 1, dtype=torch.int).to(device)
        att_src = att_src.unsqueeze(dim=0)
        att_src = att_src.repeat(batch_size_src, 1)

        logs.write_log("model input :\n" + str(src))

        action = train(model, optimizer, clip, problem_id, src, trg, att_src)
        actions[problem_id] = action.detach().cpu()
        logs.write_log("train over action: \n" + str(action))
        print(f"problem_{problem_id} inferred action: {action[0].tolist()}")

        if evaluate_test:
            get_performance(action, problem_id, eval=1)
    return actions


def train_in_one(
    problem_sequence=None,
    *,
    checkpoint_dir=None,
    evaluate_test=False,
):
    """Train one feature-conditioned policy across a problem sequence."""

    global EWC_, current_initial_populations, evaluation_round, test_evaluation_round
    EWC_ = None
    current_initial_populations = None
    evaluation_round = 0
    test_evaluation_round = 0
    print("Train In One")
    logs.write_log("Train In One")
    input_src = {}
    if problem_sequence is None:
        problem_sequence = list(continual_problem_sets[0])
    else:
        problem_sequence = [int(problem_id) for problem_id in problem_sequence]
    if not problem_sequence or any(
        problem_id < 1 or problem_id > 23 for problem_id in problem_sequence
    ):
        raise ValueError("Continual PBO problem IDs must be a non-empty list in 1..23.")

    # Compute a single embedding space for the complete declared task. Fitting
    # a new scaler at each stage changes old task vectors and invalidates EWC.
    feature = load_standardized_features(problem_sequence)
    for problem_id, vector in feature.items():
        if vector.shape[0] < d_model:
            raise ValueError(f"Problem {problem_id} has fewer than {d_model} features.")
        src = torch.from_numpy(vector[:d_model].reshape(1, d_model))
        src = src.to(torch.float32).to(device)
        input_src[problem_id] = src.repeat(batch_size_src, 1, 1)

    model, optimizer = get_model("continual")
    seen_problem_ids = []
    try:
        for stage, problem_id in enumerate(problem_sequence, start=1):
            active_problem_ids = list(dict.fromkeys(seen_problem_ids + [problem_id]))
            logs.problem_id = problem_id
            print("Train problem_" + problem_id.__str__())
            logs.write_log("Train problem_" + problem_id.__str__())
            current_initial_populations = load_initial_populations(problem_id)

            trg = torch.tensor([begin_index]).to(device)
            trg = trg.unsqueeze(dim=0)
            # train with batch
            trg = trg.repeat(batch_size_src, 1)
            att_src = torch.arange(0, 27, 1, dtype=torch.int).to(device)
            att_src = att_src.unsqueeze(dim=0)
            att_src = att_src.repeat(batch_size_src, 1)

            action = train(
                model, optimizer, clip, problem_id, input_src[problem_id], trg, att_src
            )
            logs.write_log("train over action: \n" + str(action))
            if use_ewc is True:
                print("Cal EWC______________________")
                EWC_ = EWC(model)
                for fisher_problem_id in active_problem_ids:
                    PPO_get_ewc(
                        model,
                        optimizer,
                        clip,
                        total_epoch,
                        ppo_epoch,
                        None,
                        0.2,
                        input_src[fisher_problem_id],
                        trg,
                        att_src,
                        fisher_problem_id,
                    )
                for key in EWC_._precision_matrices:
                    EWC_._precision_matrices[key] /= len(active_problem_ids)
            seen_problem_ids.append(problem_id)

            if checkpoint_dir is not None:
                output = Path(checkpoint_dir)
                output.mkdir(parents=True, exist_ok=True)
                checkpoint = output / f"stage{stage}_problem{problem_id}.pt"
                temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
                torch.save(model.state_dict(), temporary)
                temporary.replace(checkpoint)

            if evaluate_test:
                logs.write_log(
                    "EWC TEST : train over problem : " + (problem_id).__str__()
                )
                current_initial_populations = None
                for active_id in active_problem_ids:
                    with torch.no_grad():
                        old_action, _, _ = model(
                            input_src[active_id], trg, att_src, reference=True
                        )
                    logs.write_log(
                        "problem_"
                        + active_id.__str__()
                        + " action:\n"
                        + str(old_action)
                    )
                    get_performance(old_action[0:1], active_id, eval=1)
    finally:
        current_initial_populations = None


def _id_list(value):
    """Parse a comma-separated list of positive integer IDs."""

    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "IDs must be comma-separated integers."
        ) from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "At least one positive integer ID is required."
        )
    return values


def _problem_list(value):
    values = _id_list(value)
    if any(item > 23 for item in values):
        raise argparse.ArgumentTypeError("PBO problem IDs must be in 1..23.")
    return values


def _seed_list(value):
    """Parse comma-separated NumPy-compatible random seeds."""

    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Seeds must be comma-separated integers."
        ) from exc
    maximum = 2**32 - 1
    if not values or any(seed < 0 or seed > maximum for seed in values):
        raise argparse.ArgumentTypeError("Seeds must be in 0..2**32-1.")
    return list(dict.fromkeys(values))


def build_parser():
    parser = argparse.ArgumentParser(description="Train the pure-Python ALDes policy.")
    parser.add_argument(
        "--mode",
        choices=("single", "continual"),
        default=aldes_mode,
        help="single designs from scratch for each problem; continual reuses one policy",
    )
    parser.add_argument(
        "--problems",
        type=_problem_list,
        default=None,
        help="comma-separated PBO IDs (single-mode default: 1)",
    )
    parser.add_argument(
        "--seeds",
        type=_seed_list,
        default=[1],
        help="comma-separated training seeds (default: 1)",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help=(
            "run the paper's 30-run test after single training, or after every "
            "continual stage"
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="optional continual-mode state-dictionary output directory",
    )
    return parser


def main(argv=None):
    """Command-line entry point with a safe one-problem, one-seed default."""

    args = build_parser().parse_args(argv)
    if args.mode == "single" and args.checkpoint_dir is not None:
        raise SystemExit("--checkpoint-dir is only available in continual mode.")

    global EWC_, current_initial_populations, evaluation_round, test_evaluation_round
    for seed in args.seeds:
        logs.seed = seed
        logs.task_id = None
        logs.write_log(f"seed is {seed}")
        if args.mode == "continual":
            task_sequences = (
                [args.problems]
                if args.problems is not None
                else [list(problem_set) for problem_set in continual_problem_sets]
            )
            for task_index, sequence in enumerate(task_sequences, start=1):
                logs.task_id = task_index if len(task_sequences) > 1 else None
                EWC_ = None
                current_initial_populations = None
                evaluation_round = 0
                test_evaluation_round = 0
                seed_torch(seed)
                checkpoint_dir = args.checkpoint_dir
                if checkpoint_dir is not None and len(args.seeds) > 1:
                    checkpoint_dir = checkpoint_dir / f"seed{seed}"
                if checkpoint_dir is not None and len(task_sequences) > 1:
                    checkpoint_dir = checkpoint_dir / f"task{task_index}"
                train_in_one(
                    sequence,
                    checkpoint_dir=checkpoint_dir,
                    evaluate_test=args.evaluate_test,
                )
        else:
            logs.task_id = None
            EWC_ = None
            current_initial_populations = None
            evaluation_round = 0
            test_evaluation_round = 0
            seed_torch(seed)
            train_separately(args.problems or [1], evaluate_test=args.evaluate_test)


if __name__ == "__main__":
    main()
