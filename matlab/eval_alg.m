exact_datas_from_ArchSolution();
function  run_compare_alg()
alglist = ["Discrete Genetic Algorithm","Discrete Iterative Local Search","Discrete Simulated Annealing"];
alglist = ["Discrete Simulated Annealing"];
for instance =1:3
 for i = 1:numel(alglist)
     media_res=[];
     for j = 2:23
        AutoOpt('Mode','solve','Problem','pbo','InstanceSolve',[4],'AlgName',alglist(i),'ProbN',50000,'ProbFE',50,'AlgRuns',30,'problem_id',j);
        res =[];
        load(strcat(alglist(i),int2str(j)));
        [x, y] = size(bestSolutions);
        for ii =1: y
            res(ii)=cat(1,bestSolutions(1,ii).fit);
        end
        path = strcat('result/',strrep(alglist(i), ' ', '_'),'/instance_',int2str(instance),'/');
        % 确保路径存在
        if ~exist(path, 'dir')
            mkdir(path);
            addpath(genpath('result'));
        end
        fullpath = strcat(path,'f',int2str(j));
        save(fullpath,'res')
        disp(fullpath);
        media_res(j)=median(res);
     end
 end
end
end
 
function  cal_compare_mean_var()
 alglist = ["Discrete Genetic Algorithm", "Discrete Iterative Local Search","Discrete Simulated Annealing"];

 for instance = 1:3
     for i = 1:numel(alglist)
          total_mean = [];
          total_var = [];
         for j = 1:23
            path = strcat('result/',strrep(alglist(i), ' ', '_'),'/instance_',int2str(instance),'/');
            fullpath = strcat(path,'f',int2str(j));
            load(fullpath);
            total_mean(1,j) = -mean(res); %performance为了最小化问题取反，这里取反为正常值
            total_var(1,j) = var(res);
         end
              % 写入均值矩阵到 Excel 文件的第一个工作表
        filename = strcat(path,'mean_var_design_data.xlsx'); % Excel 文件名
        mean_sheet = 'Mean'; % 均值工作表名称
        xlswrite(filename, total_mean, mean_sheet);

        % 写入方差矩阵到 Excel 文件的第二个工作表
        var_sheet = 'Variance'; % 方差工作表名称
        xlswrite(filename, total_var, var_sheet);
     end
end

end
 
function  run_design_alg()
B{1}=[2,13,7,24,9,18,8];
B{2}=[2,13,4,9,19];
B{3}=[1,12,7,26,8];
B{4}=[2,11,4,9,18,8];
B{5}=[1,12,5,8];
B{6}=[0,12,7,25,8];
B{7}=[0,11,5,8];
B{8}=[2,11,5,8];
B{9}=[3,11,7,24,8];
B{10}=[1,13,7,24,8];
B{11}=[2,11,7,18,9,18];
B{12}=[2,13,7,18,9,18];
B{13}=[2,14,5,9,18];
B{14}=[1,13,7,23,9,20];
B{15}=[1,13,5,8,9,20];
B{16}=[2,14,7,19,9,19];
B{17}=[0,11,6,21,9,20];
B{18}=[2,14,4,8];
B{19}=[2,14,4,8];
B{20}=[2,12,7,26,9,18,8];
B{21}=[2,12,5,8];
B{22}=[0,12,7,25,8];
B{23}=[1,11,7,21,8];

%for another run:
% B{2}=[2, 15, 18,  7, 18,  9, 18];
% B{3}=[2, 13,  8,  4,  9, 18];
% B{4}=[ 2, 11, 10,  7, 19,  8];
% B{5}=[2, 15, 24,  4,  8];
% B{6}=[1, 11,  7, 26,  8];
% B{7}=[2, 13,  8,  5,  9, 18,];
% 
% B{8}=[2, 11,  5,  8];
% B{9}=[ 2, 13,  9, 18,  5,  8];
% B{10}=[1, 15, 21,  7, 23,  8];
% B{11}=[2, 11,  9, 18,  4,  8];
% B{12}=[2, 15, 22,  4,  9, 18];
% B{13}=[2, 14,  5,  9, 18];
% B{14}=[1, 11,  7, 25,  9, 20];
% B{15}=[2, 11,  5,  9, 19];
% B{16}=[ 0, 12,  7, 24,  9, 19];
% B{17}=[3, 15, 19,  5,  8,  9, 20];
% B{18}=[2, 15, 25,  5,  9, 19,  8];
% B{19}=[2, 14,  5,  8];
% B{20}=[2, 11,  9, 21,  7, 20];
% B{21}=[2, 11,  7, 26,  8];
% B{22}=[1, 12,  5,  8];
% B{23}=[1, 12,  7, 24,  8];

for j = 4 %1:3
        indices = [1, 13, 15, 20];
     for i = indices % 1:23
        info = ['instance:',int2str(j),' problem:',int2str(i)];
        disp(info)
        %datestr(now())
        [performance, change,solution] = get_per(B{i},i,[j],1);
        continue;
        path = ['result/design/instance_',int2str(j),'/'];
        % 确保路径存在
        if ~exist(path, 'dir')
            mkdir(path);
            addpath(genpath('result'));
        end
        fullpath = [path,'f',int2str(i)];
        res = performance(1,:);
        % save original format of solutions
        save(fullpath,'res');
        %datestr(now())
     end
end
end
 
function  cal_design_mean_var()
total_mean = [];
total_var = [];
for i = 1:3
    for j = 1:23
        path = ['result/design/instance_',int2str(i),'/f',int2str(j)];
        load(path);
        total_mean(1,j) = -mean(res);
        total_var(1,j) = var(res);
    end

    % 写入均值矩阵到 Excel 文件的第一个工作表
    filename = ['result/design/instance_',int2str(i),'/mean_var_design_data.xlsx']; % Excel 文件名
    mean_sheet = 'Mean'; % 均值工作表名称
    xlswrite(filename, total_mean, mean_sheet);

    % 写入方差矩阵到 Excel 文件的第二个工作表
    var_sheet = 'Variance'; % 方差工作表名称
    xlswrite(filename, total_var, var_sheet);
    end
end

function  run_train_in_one_design_alg()
B{1}=[1,11,7,26,8];
B{2}=[2,13,7,26,8,9,18];
B{3}=[1,11,7,26,8];
B{4}=[1,11,7,26,8];
B{5}=[1,11,7,26,8];
B{6}=[1,11,7,26,8];
B{7}=[1,11,7,26,8];
B{8}=[1,11,7,26,8];
B{9}=[3,11,7,24,8];
B{10}=[1,13,7,24,8];
B{11}=[2,11,7,18,9,18];
B{12}=[2,13,7,18,9,18];
B{13}=[2,14,5,9,18];
B{14}=[1,13,7,23,9,20];
B{15}=[1,13,5,8,9,20];
B{16}=[2,14,7,19,9,19];
B{17}=[0,11,6,21,9,20];
B{18}=[2,14,4,8];
B{19}=[2,14,4,8];
B{20}=[2,12,7,26,9,18,8];
B{21}=[2,12,5,8];
B{22}=[0,12,7,25,8];
B{23}=[1,11,7,21,8];
 for i = 1:1:8
    i
    datestr(now())
    [performance, change,solution] = get_per(B{i},i,[4],1);
    name = 'result/train_in_one'+int2str(i);
    res = performance(1,:);
    % save original format of solutions
    save(name,'res');
    datestr(now())
 end

 end

function run_autoopt_for_pbo()
AutoOpt('Mode','design','Problem','pbo','InstanceTrain',[1,2,3],'InstanceTest',4,'problem_id',1)
end
 
function exact_datas_from_ArchSolution()
    data_path = '../draw/datas/mats/ArchSolution/';
    problem_set = [1,13,15,20]
    for problem = problem_set
        path = [data_path,'ArchSolution',int2str(problem),'.mat']
        load(path);
        res = [ArchSolution(1,:).fit];
        save([data_path,'f',int2str(problem)],'res')
    end
    
end
 
