
addpath(genpath('Utilities'));
addpath(genpath('Problems'));
addpath(genpath('Main'));
addpath(genpath('Components'));

% 没开一次matlab 只执行一次
%pyversion('C:\Users\21902\.conda\envs\trrl\python.exe')

%py.IOH_Test.cosntrains()
%res = py.IOH_Test.test()
seq =[ 0, 31, 24,  7, 23, 31, 24,  9, 23, 31, 24,  6, 23, 31, 24,  8, 29, 14, 29];
problem_id = 8;
performance = get_perf(seq,problem_id,[4],1)