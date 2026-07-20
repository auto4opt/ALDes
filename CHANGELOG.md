# Changelog

All notable changes to ALDes are documented here.

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
