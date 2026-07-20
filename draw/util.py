"""Shared data-loading and plotting helpers for the paper notebooks."""

from __future__ import annotations

import copy
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt


def read_data(filename):
    pattern = re.compile(r"\d+\.\d+|\d+")
    total_steps = []
    mean_variance = []
    with Path(filename).open("r", encoding="utf-8") as stream:
        for line in stream:
            if "total env-steps" in line:
                total_steps.append(int(pattern.findall(line)[-1]))
            if "return mean" in line:
                mean_variance.append(
                    [float(number) for number in pattern.findall(line)]
                )

    means = [values[-2] for values in mean_variance]
    variances = [values[-1] for values in mean_variance]
    return total_steps, means, variances


def read_data_for_transformer(filename):
    pattern = re.compile(r"-?\d+\.\d+|-?\d+")
    data = []
    with Path(filename).open("r", encoding="utf-8") as stream:
        for line in stream:
            if "step :" not in line:
                continue
            segment = line.split("step :", maxsplit=1)[1].split("Training", maxsplit=1)[
                0
            ]
            data.append([float(number) for number in pattern.findall(segment)])
    return data


def plot_each(plotter, steps, means, deviations, label, show_stds=False):
    plotter.plot(steps, means, label=label, linewidth=2)
    if show_stds:
        plotter.fill_between(
            steps,
            [mean - deviation for mean, deviation in zip(means, deviations)],
            [mean + deviation for mean, deviation in zip(means, deviations)],
            alpha=0.2,
        )


def read_from_pkl(paths):
    data = []
    for path in paths:
        with Path(path).open("rb") as stream:
            data.append(pickle.load(stream))
    return data


def _prepare_output(save_path):
    output = Path(save_path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def roll_and_draw(frame, problem_set, save_path):
    output = _prepare_output(save_path)
    smoothed = copy.deepcopy(frame).rolling(window=5, center=False).mean()
    for problem in problem_set:
        plt.figure(figsize=(5, 3))
        plot_each(
            plt,
            smoothed["Epoch"],
            smoothed[f"Means_{problem}"],
            smoothed[f"Vars_{problem}"],
            f"problem{problem}",
            show_stds=True,
        )
        plt.title(f"F{problem}")
        plt.xlabel("Episode")
        plt.ylabel("Performance")
        plt.savefig(
            output / f"F{problem}.svg",
            dpi=300,
            format="svg",
            bbox_inches="tight",
        )


def roll_and_draw_multiple(frames, problem_set, display_names, save_path):
    output = _prepare_output(save_path)
    smoothed_frames = [
        copy.deepcopy(frame).rolling(window=5, center=False).mean() for frame in frames
    ]
    for problem in problem_set:
        plt.figure(figsize=(5, 3))
        for frame, display_name in zip(smoothed_frames, display_names):
            plot_each(
                plt,
                frame["Epoch"],
                frame[f"Means_{problem}"],
                frame[f"Vars_{problem}"],
                display_name,
                show_stds=True,
            )
        plt.title(f"F{problem}")
        plt.xlabel("Episode")
        plt.ylabel("Performance")
        plt.legend()
        plt.savefig(
            output / f"F{problem}.svg",
            dpi=300,
            format="svg",
            bbox_inches="tight",
        )


__all__ = [
    "plot_each",
    "read_data",
    "read_data_for_transformer",
    "read_from_pkl",
    "roll_and_draw",
    "roll_and_draw_multiple",
]
