import os
import ioh
import numpy as np
import array

import pandas as pd
from irace import irace
import pickle

const_problem_id = 1 
const_problem_dim = 100
const_FE = 50000
def write_excel(res,shell):
    res = res.reshape(1,-1)
    # 将 NumPy 数组转换为 pandas 数据框架
    df = pd.DataFrame(res)

    # 打开现有的 Excel 文件
    excel_file = 'D:/01Code/transformer-rl-3/mean_var_data.xlsx'
    existing_data = pd.read_excel(excel_file)

    # 将新数据追加到现有数据后面
    with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a') as writer:
        df.to_excel(writer, sheet_name=shell, index=False,header=False)
        
def tabu_pbo(x0, length, problem_id):
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
    return res

def tabu_search(problem_id,dim,problem_fe,tabu_list_length,tabu_list_cycle):
    tabu_list_length = int(tabu_list_length)
    tabu_list_cycle = int(tabu_list_cycle)
    best = np.random.randint(0,2,dim)
    best_fit = tabu_pbo(best, dim, problem_id)
    cur_problem_fe = 0
    tabu_list =  [0] * dim
    tabu_list = np.array(tabu_list)
    while(cur_problem_fe<=problem_fe):
        neighbors = build_neighbors(best,tabu_list)
        fitness = tabu_pbo(neighbors, dim, problem_id)
        if max(fitness) > best_fit:
            best_fit = max(fitness)
            best = neighbors[fitness.index(best_fit)]
            
        for i in range(len(tabu_list)):
            if tabu_list[i]>0:
                tabu_list[i] -= 1
        #计算禁忌的个数
        count_nonzero = np.count_nonzero(tabu_list)
        if count_nonzero==tabu_list_length:#禁忌表已满，减去最早加入的那个
            non_zero_elements = tabu_list[tabu_list!=0]
            min_non_zero_value = np.min(non_zero_elements)
            # 找到不为零的最小值的索引
            min_non_zero_index = np.where(tabu_list == min_non_zero_value)[0][0]
            # 将该元素置为零
            tabu_list[min_non_zero_index] = 0
        #加入最好邻域解对应的tabu
        tabu_list[fitness.index(max(fitness))] = tabu_list_cycle
        cur_problem_fe = cur_problem_fe + len(neighbors)
    print('problem{1},dim {2},best fitness {0} '.format(best_fit,problem_id,dim))
    return  best_fit

def build_neighbors(X0,tabu_list):
    neighbors = []
    for i in range(X0.shape[0]):
        if tabu_list[i]==0:
            X_temp = X0.copy()
            X_temp[i] = int(not(X_temp[i]))
            neighbors.append(X_temp)
    return neighbors

def run_tabu():
    res = np.zeros((30, 23))
    for j in range(30):
        for i in range(1,24):
            res[j,i-1] = tabu_search(i,625,50000,625,3)

    column_means = np.mean(res, axis=0)

    # 计算每行的方差
    column_variances = np.var(res, axis=0)
    write_excel(column_means,'Tabu_mean')
    write_excel(column_variances, 'Tabu_var')
    print("tsa")
    
    
# This target_runner is over-complicated on purpose to show what is possible.
def target_runner(experiment, scenario):
    if scenario['debugLevel'] > 0:
        # Some configurations produced a warning, but the values are within the limits. That seems a bug in scipy. TODO: Report the bug to scipy.
        print(f'{experiment["configuration"]}')
    res = tabu_search(const_problem_id,const_problem_dim,const_FE,**experiment['configuration'])

    return dict(cost=-res)

    
parameters_table = '''
tabu_list_length       "" i (1, 625)
tabu_list_cycle        "" i (1, 100) 
'''

default_values = '''
tabu_list_length tabu_list_cycle 
10               3            
'''

# These are dummy "instances", we are tuning only on a single function.
instances = np.arange(100)

# See https://mlopez-ibanez.github.io/irace/reference/defaultScenario.html
scenario = dict(
    instances = instances,
    maxExperiments = 100,
    debugLevel = 3,
    digits = 5,
    parallel=10, # It can run in parallel ! 
    logFile = "")

def run_irace():
    tuner = irace(scenario, parameters_table, target_runner)
    tuner.set_initial_from_str(default_values)
    best_confs = tuner.run()
    
    total_res = []
    # Pandas DataFrame
    print(best_confs)
    #for i in range(len(best_confs)):
    #only use the first 
    #print(f'tabu_list_length:{best_confs['tabu_list_length'][0]}, tabu_list_cycle:{best_confs['tabu_list_cycle'][0]}')
    for epoch in range(30):
        res = tabu_search(const_problem_id,const_problem_dim,const_FE,best_confs['tabu_list_length'][0],best_confs['tabu_list_cycle'][0])
        total_res.append(res)
    return total_res


def write_excel(res,shell,path):
    # 将 NumPy 数组转换为 pandas 数据框架
    df = pd.DataFrame(res)

    # 打开现有的 Excel 文件
    existing_data = pd.read_excel(path)

    # 将新数据追加到现有数据后面
    with pd.ExcelWriter(path, engine='openpyxl', mode='a') as writer:
        df.to_excel(writer, sheet_name=shell, index=False,header=False)            
            
df = pd.DataFrame(columns=['Problem', 'dim', 'Mean', 'Variance'])
df.to_csv('../matlab/result/tabu_irace/instance_4/result.csv', index=False)
#df = pd.read_csv('/home/booze/code/ALDes/tabu_result.csv')     
for _dim in [625]:
    for _problem_id in range(1,24):
        const_problem_dim = _dim
        const_problem_id = _problem_id
        print(f'current problem dim,id: {const_problem_dim} , {const_problem_id}')
        res = run_irace()

        dump_file = f'../matlab/result/tabu_irace/instance_4/f{_problem_id}.pkl'
        with open(dump_file, 'wb') as f:
            # 使用pickle.dump()将字典对象序列化并保存到文件中
            pickle.dump(res, f)

        means = np.mean(res)
        variances = np.var(res)

        df = df._append({
            'Problem': _problem_id,
            'dim': _dim,
            'Mean': means,
            'Variance': variances
        }, ignore_index=True)

        df.to_csv('../matlab/result/tabu_irace/instance_4/result.csv', index=False)