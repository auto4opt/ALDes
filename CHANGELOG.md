# Changelog

All notable changes to ALDes are documented here.

## 2.0.1 - 2026-09-02

### Fixed

- Kept continual-task feature embeddings fixed across stages instead of
  repeatedly rescaling and changing all previously seen problem vectors.
- Run the paper's two default continual tasks from separate initial policies
  and made the expensive stage-by-stage test protocol explicitly opt-in.
- Prevented 100/225/400-dimensional training populations from being supplied
  to the 625-dimensional test instance.
- Implemented the empirical diagonal Fisher as the mean of per-algorithm
  squared score gradients and restored the `lambda / 2` EWC coefficient from
  the paper.
- Kept the final annealed PPO update at a non-zero learning rate and made
  independent `(problem, seed)` trials invariant to CLI problem order.
- Isolated training and test random streams so enabling the continual
  forgetting-matrix evaluation cannot change later policy updates.
- Matched continual conditioning to the paper's prepended feature-token
  architecture, restored causal attention behavior, and added clear
  validation for invalid model and replay inputs.
- Validated feature and initial-population artifacts, preserved failed
  experiment status, wrote histories/checkpoints atomically, recorded
  dependency versions, and corrected scientific notation and signed-number
  parsing in plotting helpers.
- Corrected PBO maximization direction, Hamming nearest-better semantics, and
  the exact Table A3 factor schema for continual landscape features. The
  paper's `100 * dimension` sampling factor is the default, including for
  information-content distances. Pairwise statistics are blockwise and
  high-dimensional walks retain only sampled populations, avoiding quadratic
  distance matrices and infeasible repeated feature regressions.
- Restricted the AutoOptLib dependency to the audited 1.3 feature/protocol
  series so a future minor release cannot silently change experiment
  semantics.
- Included citation metadata, the changelog, plotting notebooks, and their
  historical input artifacts in source distributions used for reproducibility.
- Required the release workflow to pass linting and tests before publishing,
  and covered every advertised Python minor version in CI.

## 2.0.0 - 2026-07-20

### Added

- A pure-Python execution path backed by AutoOptLib 1.3.0, with no MATLAB or
  MATLAB Engine runtime dependency.
- Automatic PyTorch accelerator selection for NVIDIA CUDA, AMD ROCm, Apple
  MPS, and CPU, with environment-variable overrides.
- Deterministic CPU multiprocessing for generated-algorithm and objective
  evaluation while neural-network work remains on the selected PyTorch device.
- Explicit independent single-problem and feature-conditioned continual-design
  modes. Independent design is the default and does not extract problem
  features; continual design uses landscape features and EWC.
- A constrained generator and matching executor grammar: fork can follow
  choose, each branch contains one search operation except that crossover may
  be followed by mutation, and branches merge into one shared update.
- Command-line entry points for training and the time-bounded paper subset,
  portable packaged reference results, tests, and cross-version CI.

### Changed

- Aligned the PBO training protocol with the paper: 100 PPO epochs, 16 sampled
  algorithms per epoch, five PPO updates, 5,000 training evaluations, and an
  optional 30-run/50,000-evaluation final test.
- Replaced the historical Python-to-MATLAB bridge with the public Python
  AutoOptLib ALDes backend and its 32-token vocabulary.
- Made one PBO problem and one seed the safe default command-line scope.
- Moved historical plotting results from the removed MATLAB tree into
  `draw/datas/reference_results` without changing their contents.

### Removed

- Bundled MATLAB source, MATLAB-specific bridge scripts, stale bytecode,
  obsolete TorchText data loaders, and generated experiment artifacts.

## 1.0.0

- Original research release combining Python model training with MATLAB
  algorithm execution.
