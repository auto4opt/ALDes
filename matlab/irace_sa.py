import os
import ioh
import numpy
import numpy as np
import array
import random
import pandas as pd
from irace import irace
import math
import matplotlib.pyplot as plt
import socket
import pickle

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 设置接收数据的超时时间为 5 秒
sock.settimeout(600)
    # 连接到服务器
server_address = ('192.168.1.109', 30000)
sock.connect(server_address)

const_problem_id = 1 
const_problem_dim = 1 # for matlab dim 1,2,3 for train,4 for test [100 225 400 625] 
dim_to_normal = [100, 225, 400, 625] 
const_FE = 50000

random.seed(0)

parameters_table = '''
T         "" r (0.01, 0.05)
'''

default_values = '''
T           
0.03           
'''

# These are dummy "instances", we are tuning only on a single function.
instances = np.arange(100)

# See https://mlopez-ibanez.github.io/irace/reference/defaultScenario.html
scenario = dict(
    instances = instances,
    maxExperiments = 200,
    debugLevel = 3,
    digits = 5,
    parallel=1, # It can run in parallel ! 
    logFile = "")
        
    
def socket_sa(T):
    try:
        message = f"{T},{const_problem_id},{const_problem_dim}"
        # 发送指令给 MATLAB 服务器
        sock.sendall(message.encode())

        # 等待并接收响应
        data = sock.recv(1024)
        print('Received:', data.decode())

    except socket.timeout:
        print('Socket timed out while waiting for response')
    return float(data.decode())

def sa_pbo(x0, length, problem_id):
    #print("test")
    #print(int(problem_id))
    # In order to instantiate a problem instance, we can do the following:
    problem = ioh.get_problem(
        int(problem_id),
        instance=1,
        dimension=int(length),
        problem_class=ioh.ProblemClass.PBO
    )

    problem.enforce_bounds(how=ioh.ConstraintEnforcement.SOFT, weight=1.0, exponent=1.0)

    # We can access the contraint information of the problem
    population = np.array(x0).astype(int)
    #print(population)
    # Evaluation happens like a 'normal' objective function would
    res = problem(population)
    neg_res = [-x for x in res]
    return neg_res


def sa(problem_id,dim,problem_fe,T):
    problem_fe = 50000
    prob_N = 50
    Tf = 0.01
    Gmax = problem_fe/prob_N

    cur_problem_fe = 0
    # calculate iters
    alpha  = (Tf/T)**(1/Gmax)
    now_x = []
    for i in range(prob_N):
        x = np.random.randint(0,2,dim)
        now_x.append(x)
    best = 0
    current_fe = 0
    G = 0
    record = [] # record each iter best
    record_new = [] # record each iter best
    # annealing
    while T > Tf and cur_problem_fe < problem_fe:
        # iteration
        f_now = sa_pbo(now_x, dim, problem_id)
        record.append(min(f_now))
        new_x = get_new_x(now_x)
        f_new = sa_pbo(new_x, dim, problem_id)
        record_new.append(min(f_new))        
        accept = Metrospolis(T,f_now, f_new)

        for i in range(len(accept)):
            if random.random() < accept[i]:
                now_x[i]=new_x[i]
        if min(f_new)<best:
            best = min(f_new)
            #print(f'G:{G},best:{best}')

        # cooling
        T = T * alpha
        current_fe = current_fe + len(now_x)
        G = G+1

    # plt.plot(record)
    # plt.plot(record_new)
    # plt.savefig('plot.png')
    print(best)
    return best 

def Metrospolis(T, f_now, f_new):
    accept = []
    for i in range(len(f_now)):
        if f_new[i] < f_now[i]:
            accept.append(1)
        else:
            p = math.exp((f_now[i]-f_new[i]) /abs(f_now[i]+1e-6) / T)
            accept.append(p)
    
    return accept
def get_new_x(X0):
    X_temp = X0.copy()
    for i in range(len(X0)):
        random_integer = random.randint(0, X0[i].shape[0]-1)
        X_temp[i][random_integer] = int(not(X_temp[i][random_integer]))
    return X_temp

    

# test
def test_default_set():
    df = pd.DataFrame(columns=['Problem', 'dim', 'Mean', 'Variance'])
    for _problem_id in range(1,24):
        total_res=[]
        for epoch in range(30):
            res = sa(_problem_id,625,const_FE,0.1)
            total_res.append(res)    

        means = np.mean(total_res)
        variances = np.var(total_res)  
        df = df._append({
        'Problem': _problem_id,
        'dim': 625,
        'Mean': means,
        'Variance': variances
        }, ignore_index=True)
    return


# This target_runner is over-complicated on purpose to show what is possible.
def target_runner(experiment, scenario):
    if scenario['debugLevel'] > 0:
        # Some configurations produced a warning, but the values are within the limits. That seems a bug in scipy. TODO: Report the bug to scipy.
        print(f'{experiment["configuration"]}')
        pass
    #res = sa(const_problem_id,const_problem_dim,const_FE,**experiment['configuration'])
    res = socket_sa(**experiment['configuration'])
    return dict(cost=res)

    

def run_irace():
    tuner = irace(scenario, parameters_table, target_runner)
    tuner.set_initial_from_str(default_values)
    best_confs = tuner.run()
    total_res = []
    print(f'best_cofs:{best_confs}')
    
    for epoch in range(30):
        res = socket_sa(best_confs['T'][0])
        #res = sa(const_problem_id,const_problem_dim,const_FE,best_confs['T'][0])
        total_res.append(res)
    return total_res

#df = pd.read_csv('/home/booze/code/ALDes/sa_results.csv')          
df = pd.DataFrame(columns=['Problem', 'dim', 'Mean', 'Variance'])
df.to_csv('matlab/result/sa_irace/instance_4/result.csv', index=False)
for _dim in [4]:
    for _problem_id in range(1,24):
        const_problem_id = _problem_id
        const_problem_dim = _dim
        res = run_irace()
        dump_file = f'matlab/result/sa_irace/instance_4/f{_problem_id}.pkl'
        with open(dump_file, 'wb') as f:
            # 使用pickle.dump()将字典对象序列化并保存到文件中
            pickle.dump(res, f)
        means = np.mean(res)
        variances = np.var(res)
        df = df._append({
            'Problem': _problem_id,
            'dim': dim_to_normal[_dim-1],
            'Mean': means,
            'Variance': variances
        }, ignore_index=True)
        df.to_csv('matlab/result/sa_irace/instance_4/result.csv', index=False)


