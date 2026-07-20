import os

from util.device import resolve_device

# Automatically prefer NVIDIA CUDA / AMD ROCm, then Apple MPS, and finally
# CPU. Set ALDES_DEVICE=cpu/cuda/mps/rocm to override automatic selection.
device_preference = os.environ.get("ALDES_DEVICE", "auto")
device = resolve_device(device_preference)

# Transformer architecture
max_len = 50
d_model = 32
n_layers = 8
n_heads = 8
ffn_hidden = 2048
drop_prob = 0.1

# ALDes has two explicit modes. Single-problem design is the default and does
# not extract or feed landscape features. Continual design conditions the
# policy on paper-style PBO features and may use EWC between tasks.
aldes_mode = "single"  # "single" or "continual"
use_ewc = True
ewc_weight = 200.0
continual_problem_sets = (
    (1, 2, 3, 4, 5, 6, 7),
    (1, 2, 11, 18, 19, 22, 23),
)

# Optimizer and PPO settings
init_lr = 5e-5
adam_eps = 5e-9
clip = 1.0
weight_decay = 5e-4
total_epoch = 100
ppo_epoch = 5
batch_size_src = 16
