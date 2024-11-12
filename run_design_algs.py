import os

import matlab.engine
import matlab

import numpy as np
import re
import ray
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pandas as pd

eng = matlab.engine.start_matlab()
# 此地址为test.m文件存放的地址
work_path = os.getcwd() + "\\matlab"
eng.cd(work_path)

def run(alg):
    action = alg.action
    problem_id = alg.problem_id
    seed = alg.seed
    save_path = alg.save_path
    # delet first one "Begin"
    action.pop(0)
    # delet "End"
    while (action[len(action) - 1] == 17):
        action.pop()  # delet last one

    alg  = matlab.double(initializer=action)

    # default eval setting
    instances = matlab.double([4])
    eval = 1

    [performance, change,solution] = eng.get_per(alg, problem_id, instances, eval, nargout=3)

    performance = np.array(performance)
    performance = performance[0]

    dump_file = save_path + f'seed{seed}_problem{problem_id}_res.pkl'
    with open(dump_file, 'wb') as f:
        # 使用pickle.dump()将字典对象序列化并保存到文件中
        pickle.dump(performance, f)
    return performance

class Alg:
    def __init__(self, action, problem_id, seed, path):
        self.action = action
        self.problem_id = problem_id
        self.seed = seed
        self.save_path = path
    
def read_algs(path, seeds, problem_set):
    total_algs = []
    pattern = re.compile(r'\d+\.\d+|\d+')

    log_file = path + 'log.txt' 
    with open(log_file, 'r', encoding='utf-8') as f:
        keyword = 'train over action:'
        lines = f.readlines()

        index = 0
        for seed in seeds:
            for problem_id in problem_set:
                while index < len(lines):
                    line = lines[index]
                    if keyword in line:
                        index += 1
                        numbers_of_alg = pattern.findall(lines[index])
                        # convert str list to int list
                        numbers_of_alg = list(map(int, numbers_of_alg))

                        alg = Alg(numbers_of_alg, problem_id,seed,path)
                        total_algs.append(alg)
                        break
                    index += 1 
 
    return total_algs

# ray sames not suit with matlab
def ray_parral():
    num_workers = 8
    while True:
        num_runs_left = len(total_algs)
        num_processes = min(num_workers, num_runs_left)

        total_works = []
        for _ in range(num_processes):
            alg = total_algs.pop()
            total_works.append(run.remote(alg.alg, alg.problem_id, alg.seed, save_path))

        # collect results
        outputs = ray.get(total_works)
    

def run_parral(seeds, problem_set, path):
    seeds = [1, 2, 3, 4, 5]
    problem_set = [1, 3, 14, 15, 17, 20]

    total_algs = read_algs(path, seeds, problem_set)

    #total_algs = total_algs[-6:] # run seed5 algs
    # 创建一个包含3个线程的线程池
    with ThreadPoolExecutor(max_workers=6) as executor:
        # 提交任务到线程池
        futures = [executor.submit(run, alg) for alg in total_algs]

        # 使用 as_completed 方法获取任务结果
        for future in as_completed(futures):
            result = future.result()
            print(result)

def load_pkls(path, seeds, problem_set):        
    df = pd.DataFrame(columns=['Problem', 'seed', 'dim', 'Mean', 'Variance'])
    df.to_csv(path + 'result.csv', index=False)

    for problem in problem_set:
        datas = []
        for seed in seeds:
            file_name = f'seed{seed}_problem{problem}_res.pkl'
            file_path = path + file_name

            with open(file_path, 'rb') as f:
                # loaded_data: steps(100)*algs(16)*instance(3)*runs(5)
                data = pickle.load(f)
                datas.append(data)

        datas = np.array(datas)
        mean = -datas.mean()
        var = datas.var()

        df = df._append({
            'Problem': problem,
            'seed': seed,
            'dim': 625,
            'Mean': mean,
            'Variance': var
        }, ignore_index=True)
        df.to_csv(path + 'result.csv', index=False)
if __name__ == '__main__':

    seeds = [1, 2, 3, 4, 5]
    seeds = [1]
    problem_set = [1, 3, 14, 15, 17, 20]
    FE3000_path = 'D:\\01Code\\ALDes\\draw\\datas\\pkls\\3000FE\\'
    FE10000_path = 'D:\\01Code\\ALDes\\draw\\datas\\pkls\\10000FE\\'

    #run_parral(seeds, problem_set,FE3000_path)

    load_pkls(FE3000_path, seeds, problem_set)


