import torch

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model parameter setting
batch_size = 1
max_len = 50
d_model = 32
n_layers = 8
n_heads = 8
ffn_hidden = 2048
drop_prob = 0.1

# optimizer parameter setting
init_lr = 5e-5
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100
epoch = 1000
clip = 1.0
weight_decay = 5e-4
inf = float('inf')

# my config
total_epoch = 50 # DEFULT 100
ppo_epoch = 5
batch_size_src = 16

model_type = 0 # 0 :transformer 1: LSTM
train_type = 0 # 0 :PPO 1: PG

