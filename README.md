# ALDes

Pure-Python implementation of **Automated Metaheuristic Algorithm Design with
Autoregressive Learning** (ALDes).

ALDes treats algorithm design as constrained autoregressive sequence generation.
A Transformer policy generates a variable-length metaheuristic program, PPO
learns from the program's performance, and AutoOptLib executes the generated
algorithm without MATLAB or MATLAB Engine.

## Release scope

This release supports the paper's 23 pseudo-Boolean optimization (PBO) tasks:

- independent design from scratch for one target problem;
- feature-conditioned continual design with EWC;
- the paper's training and test budgets;
- automatic neural-network acceleration on CUDA, ROCm, or Apple MPS;
- parallel CPU evaluation of generated algorithms.

The paper's RIS beamforming and power-system restoration experiments are not
part of this release. The repository therefore makes no claim that those two
application results can be reproduced here.

## Installation

Python 3.9--3.11 is supported. Install the released package from PyPI:

```bash
python -m pip install aldes
```

For an editable source installation, clone this repository and install it from
the repository root:

```bash
git clone --branch v2.0.0 --depth 1 https://github.com/auto4opt/ALDes.git
cd ALDes
python -m pip install -e .
```

The installation automatically downloads the compatible
`autooptlib[aldes]` dependency from PyPI.
Users do not need to clone AutoOptLib separately. The Python import name is
lowercase:

```python
import autooptlib
```

A Conda environment can be created instead:

```bash
conda env create -f aldes.yaml
conda activate aldes
```

For development and tests, install the test extra:

```bash
python -m pip install -e ".[test]"
```

## Quick start

The default command performs one independent design trial on PBO F1 with
seed 1:

```bash
aldes-train
```

The equivalent source command is:

```bash
python train.py
```

The inferred algorithm sequence is printed and written to the run log. A
single-problem run does not save a model checkpoint because the trained policy
is not reused after its final algorithm has been inferred.

Choose another problem or multiple explicit trials with command-line options:

```bash
aldes-train --problems 14 --seeds 1
aldes-train --problems 1,14,15 --seeds 1,2
aldes-train --problems 14 --seeds 1 --evaluate-test
```

`--evaluate-test` applies the paper's full 30-run test protocol after training.
Without it, only training and final algorithm inference are performed.

## Continual design

Continual mode extracts PBO landscape features, conditions one policy on those
features, and applies EWC between tasks:

```bash
aldes-train --mode continual --problems 1,2,11 --seeds 1
```

The default continual sequence is defined in `conf.py`. Checkpoint output is
optional and is only intended for a policy that must be reused across continual
stages:

```bash
aldes-train --mode continual --checkpoint-dir logs/continual
```

No checkpoint binary is distributed with this repository.

## Compute devices

PyTorch devices are selected automatically in this order:

1. NVIDIA CUDA or AMD ROCm;
2. Apple Metal Performance Shaders (MPS);
3. CPU.

PyTorch must have been built for the user's accelerator. In ROCm builds, AMD
devices are exposed through PyTorch's `cuda` API. Override automatic selection
when needed:

```bash
ALDES_DEVICE=cpu aldes-train
ALDES_DEVICE=cuda aldes-train
ALDES_DEVICE=mps aldes-train
```

Only neural-network training and inference use the selected accelerator.
Generated algorithms and objective functions run on CPUs. Candidate algorithms
are evaluated in parallel using up to the available logical CPU cores:

```bash
ALDES_EVAL_WORKERS=8 aldes-train
ALDES_EVAL_WORKERS=1 aldes-train  # disable multiprocessing
```

## Paper protocol

The default configuration for each PBO design trial is:

- training instances: dimensions 100, 225, and 400;
- 100 PPO epochs and 16 generated algorithms per epoch;
- 5 PPO updates per epoch;
- 5 runs per training instance and 5,000 function evaluations per run;
- population size 50;
- test instance: dimension 625;
- 30 test runs and 50,000 function evaluations per run.

The single-problem mode does not extract or input problem features. Feature
conditioning is enabled only in continual mode.

## Time-bounded paper subset

The repository includes a runner that completes whole paper-protocol trials
until the next problem is predicted to exceed a wall-clock budget:

```bash
aldes-paper-subset --problems 1,14,15 --time-budget-minutes 60
```

The budget is checked only between problems. A started problem always retains
all 100 epochs and the complete 30-run test. Structured JSON results are written
under `experiments/`; this local output directory and all model checkpoints are
ignored by Git.

Historical comparison data used by the plotting notebooks are stored under
`draw/datas/reference_results` and can be read with SciPy.

## Tests

Run the test suite and static checks with:

```bash
python -m pytest -q
python -m ruff check .
```

GitHub Actions repeats installation, linting, and tests on Linux with Python
3.9 and 3.11. Tests cover device selection, serial/parallel evaluation parity,
single and continual feature modes, grammar-valid generation, PPO likelihood
replay, EWC accumulation, and a pure-Python PPO update.

## Citation

If you use ALDes, cite:

> Q. Zhao, T. Liu, B. Yan, Q. Duan, J. Yang, and Y. Shi, "Automated
> Metaheuristic Algorithm Design with Autoregressive Learning," IEEE
> Transactions on Evolutionary Computation, 2024.
> https://doi.org/10.1109/TEVC.2024.3464677

## License

ALDes is released under the Apache License 2.0. See `LICENSE`.
