import os

import ioh
import numpy
import numpy as np
import array

import pandas as pd

def test(x0):
    print('go!')
    x0 = np.array(x0)
    # In order to instantiate a problem instance, we can do the following:
    problem = ioh.get_problem(
        "Sphere",
        instance=1,
        dimension=10,
        problem_class=ioh.ProblemClass.REAL
    )
    # We can access the contraint information of the problem
    #x0 = np.random.uniform(problem.bounds.lb, problem.bounds.ub,[10,10])


    # Evaluation happens like a 'normal' objective function would
    return problem(x0)

def PBO(x0,length,problem_id):
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
    ret = array.array('d', res)
    return ret

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

def BBOB(x0,length,problem_id):
    #print("test")
    #print(int(problem_id))
    #print(length)
    # In order to instantiate a problem instance, we can do the following:
    problem = ioh.get_problem(
        int(problem_id),
        instance=1,
        dimension=int(length),
        problem_class=ioh.ProblemClass.BBOB
    )

    #problem.enforce_bounds(how=ioh.ConstraintEnforcement.SOFT, weight=1.0, exponent=1.0)

    # We can access the contraint information of the problem
    population = np.array(x0).astype(float)
    #print(population)
    # Evaluation happens like a 'normal' objective function would
    res = problem(population)
    #print(res)
    ret = array.array('d', res)
    return ret
def cosntrains():
    # We take the Shere function as example
    p = ioh.get_problem("Sphere", 1, 2)
    # There a several strategies that modify a constraint's behavoir:
    types = (
        ioh.ConstraintEnforcement.NOT,       # Don't calulate the constraint function
        ioh.ConstraintEnforcement.HIDDEN,    # Calulate the constraint, but don't penalize
        ioh.ConstraintEnforcement.SOFT,      # Calculate both constraint and objective function value y, and penalize as y + p
        ioh.ConstraintEnforcement.HARD,      # Calulate the constraint, if there is violation, don't calculate y, and only return p
    )
    for strategy in types:
        p.enforce_bounds(how=strategy, weight=1.0, exponent=1.0)
        print(strategy, p([7, 7]), p.constraints.violation())
def test():
    problem = ioh.get_problem(
        1,
        instance=1,
        dimension=2,
        problem_class=ioh.ProblemClass.BBOB
    )

    #problem.enforce_bounds(how=ioh.ConstraintEnforcement.SOFT, weight=1.0, exponent=1.0)

    while(problem.state.optimum_found!=True):
        xO = np.random.random(size=[1000,2])*10-5
        xO = [-1.63514,1.14487]
        #res = BBOB(xO, 2, 1)
        res = problem(xO)
        print(res)
        #print(problem.state.current_best)
        if problem.state.optimum_found ==True:
            print(res)
            break
def checkoptimum():
    #path =os.getcwd()
    #print(os.path.abspath(os.path.dirname(os.getcwd())))
    f = open(os.path.dirname(os.getcwd()) + '/result/bbob_optimun' + '.txt', 'a')

    for i in range(1,25):
        for j in [2,5,10,20]:
            problem = ioh.get_problem(
                i,
                instance=1,
                dimension=j,
                problem_class=ioh.ProblemClass.BBOB
            )
            print('id ={},dim={},optimum={} \n'.format(i,j,problem.optimum))
            f.write('id ={},dim={},optimum={} \n'.format(i,j,problem.optimum))

    f.close()

def tabu_search(tabu_list_length,tabu_list_cycle,problem_id,dim,problem_fe):
    best = np.random.randint(0,2,dim)
    best_fit = tabu_pbo(best, dim, problem_id)
    cur_problem_fe = 0
    tabu_list =  [0] * tabu_list_length
    while(cur_problem_fe<=problem_fe):
        neighbors = build_neighbors(best,tabu_list)
        fitness = tabu_pbo(neighbors, dim, problem_id)
        if max(fitness) > best_fit:
            best_fit = max(fitness)
            best = neighbors[fitness.index(best_fit)]
        for i in range(len(tabu_list)):
            if tabu_list[i]>0:
                tabu_list[i] -= 1
        tabu_list[fitness.index(max(fitness))] = tabu_list_cycle
        cur_problem_fe = cur_problem_fe + len(neighbors)
    print('problem{1},best fitness {0} '.format(best_fit,problem_id))
    return  best_fit

def build_neighbors(X0,tabu_list):
    neighbors = []
    for i in range(X0.shape[0]):
        if tabu_list[i]==0:
            X_temp = X0.copy()
            X_temp[i] = int(not(X_temp[i]))
            neighbors.append(X_temp)
    return neighbors
# xO = np.random.randint(0,2,64)
# length = 64
# PBO(xO,length,2)
#test()
#checkoptimum()
def run_tabu():
    res = np.zeros((30, 23))
    for j in range(30):
        for i in range(1,24):
            res[j,i-1] = tabu_search(625,3,i,625,50000)

    column_means = np.mean(res, axis=0)

    # 计算每行的方差
    column_variances = np.var(res, axis=0)
    write_excel(column_means,'Tabu_mean')
    write_excel(column_variances, 'Tabu_var')
    print("tsa")

def write_excel(res,shell,file_path):
    res = res.reshape(1,-1)
    # 将 NumPy 数组转换为 pandas 数据框架
    df = pd.DataFrame(res)

    # 打开现有的 Excel 文件
    existing_data = pd.read_excel(file_path)

    # 将新数据追加到现有数据后面
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
        df.to_excel(writer, sheet_name=shell, index=False,header=False)
#run_tabu()