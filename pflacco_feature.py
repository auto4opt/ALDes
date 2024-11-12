from datetime import datetime
import pandas as pd
from pflacco.classical_ela_features import *
from pflacco.sampling import create_initial_sample
from ioh import get_problem, ProblemType
import ioh
from sklearn.preprocessing import StandardScaler


def test_pflacco():
    features = []
    # Get all 24 single-objective noiseless BBOB function in dimension 2 and 3 for the first five instances.
    for fid in range(1,25):
        for dim in [2, 3]:
            for iid in range(1, 6):
                # Get optimization problem
                #problem = get_problem(fid, iid, dim, problem_type = ProblemType.BBOB)
                problem = ioh.get_problem(
                    fid,
                    instance=iid,
                    dimension=dim,
                    problem_class=ioh.ProblemClass.BBOB
                )
                # Create sample
                X = create_initial_sample(dim, lower_bound = -5, upper_bound = 5)
                y = X.apply(lambda x: problem(x), axis = 1)

                # Calculate ELA features
                ela_meta = calculate_ela_meta(X, y)
                ela_distr = calculate_ela_distribution(X, y)
                ela_level = calculate_ela_level(X, y)
                nbc = calculate_nbc(X, y)
                disp = calculate_dispersion(X, y)
                ic = calculate_information_content(X, y, seed = 100)

                # Store results in pandas dataframe
                data = pd.DataFrame({**ic, **ela_meta, **ela_distr, **nbc, **disp, **{'fid': fid}, **{'dim': dim}, **{'iid': iid}}, index = [0])
                features.append(data)

    features = pd.concat(features).reset_index(drop = True)

    print(features)

# input solution: x-y-x-y-...
def cal_feature(solution):
    for i in range(0,len(solution),2):
        x = pd.DataFrame(np.array(solution[i])[0:10,:])
        y= (np.array(solution[i+1])[0:10,:]).squeeze()
        ela_meta = calculate_ela_meta(x, y)
    return ela_meta

        # x:10*200--》60s
        #   10*400-->1088s


def cal_PBO_feature(problem_id,instance,dim,seed):
    print("\nbegin_{0}_{1}_{2}_{3}_".format(problem_id,instance,dim,seed)+ datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    X = np.random.randint(0, 2, [10*dim,dim])
    problem = ioh.get_problem(
        problem_id,
        instance=instance,
        dimension=dim,
        problem_class=ioh.ProblemClass.BBOB
    )
    Y = problem(X)

    x = pd.DataFrame(X)
    y = (np.array(Y)).squeeze()
    nbc = calculate_nbc(x, y)
    ic = calculate_information_content(x, y, seed=100)
    ela_meta = calculate_ela_meta(x, y)
    #ela_distr = calculate_ela_distribution(x, y)
    #ela_level = calculate_ela_level(x, y)

    disp = calculate_dispersion(x, y,dist_method = 'hamming')

    data = pd.DataFrame({**ic, **ela_meta, **nbc, **disp, **{'fid': problem_id}, **{'dim': dim}, **{'iid': instance}, **{'seed': seed}},
                        index=[0])
    #print(data)
    print("end" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return data
#test_pflacco()
def save_ela():
    medians = []
    for problem in range(1,24,1):
        features = []
        for dim in [100]:
            for seed in range(1,6,1):
                data = cal_PBO_feature(problem,1,dim,seed)
                features.append(data)

        features = pd.concat(features).reset_index(drop = True)
        median = pd.DataFrame(features.median())
        median.T.to_csv("ela/ela_result{0}.csv".format(problem))
    #medians = pd.concat(medians).reset_index(drop=True)
    #medians.to_csv("ela_result.csv")



def transform():
    elas = []
    for problem in range(1,24,1):
        ela = pd.read_csv("ela/ela_result{0}.csv".format(problem), index_col=0)
        elas.append(ela)
    elas = pd.concat(elas).reset_index(drop=True)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(elas)

    feature = X_train[:,:-4]
    return feature
