import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from datetime import datetime
import os
import numpy as np
import pickle

def _get_action(model_output):
    # 偶数列
    mean = model_output[:, 1::2]
    # 奇数列
    sdv = model_output[:, 0::2]
    # here use exp make sure its positive
    sdv = torch.exp(sdv)

    probs = Normal(mean, sdv)
    action = probs.sample()
    action = torch.sigmoid(action)
    return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)

def seed_torch(seed=2): #1029
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
     
class my_log():
    def __init__(self):
        now = datetime.now()  # 获得当前时间
        global_timestr = now.strftime("%m_%d_%H_%M")
        self.log_dir = f'logs/{global_timestr}/'
        os.makedirs(self.log_dir)
        self.log_file = self.log_dir + 'log.txt'
        
        self.problem_id = -1
        self.seed = -1

    
    def write_log(self,info):
        f = open(self.log_file, 'a')
        now = datetime.now()  # 获得当前时间
        timestr = now.strftime("%m_%d_%H_%M")
        f.write('\n' + timestr + ":"+info)
        f.close()

    def dump_log(self,experience):
        dump_file = self.log_dir + f'seed{self.seed}_problem{self.problem_id}_.pkl'
        with open(dump_file, 'wb') as f:
            # 使用pickle.dump()将字典对象序列化并保存到文件中
            pickle.dump(experience, f)
         

