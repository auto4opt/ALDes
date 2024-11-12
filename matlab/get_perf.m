function performance = get_perf(seq,problem_id,instanceTrain,eval)
% get the performance of the designed algorithm

addpath(genpath('Utilities'));
addpath(genpath('Problems'));
addpath(genpath('Main'));
addpath(genpath('Components'));
addpath(genpath('C:\Program Files\MATLAB\R2020b\toolbox\matpower7.1'));
warning('off')

% seq = [1.0,29.0, 9.0,19.0,29.0, 12.0,29.0]; % simple one search operator
% seq = [2.0,29.0, 5.0,29.0, 10.0,20.0,29.0, 13.0,29.0]; % simple GA
% seq = [3.0,31.0,21.0, 8.0,29.0, 11.0,29.0, 14.0,29.0]; % with fork
% seq = [0.0,29.0, 8.0,30.0,21.0, 11.0,29.0, 16.0,19.0,29.0]; % with iterate
% seq = [3.0,31.0,20.0,8.0,29.0,12.0,29.0]; 

% instanceTrain = 1;
% problem_id = 0;
% eval = 0;

% setting
problem      = 'pbo'; % target problem, pbo/beamforming/blackstart
seq_max_comp = 6; % maximun number of components in a sequence
num_comp     = 17; % number of candidate components
other_token  = {'begin';'end';1;2;3;4;5;6;7;8;9;10;'forward';'iterate';'fork'}; % tokens that not involved in the matlab AutoOpt software
values       = {0;0;0;0; 0;0;1;1;0;1;1;0; 0;0;0;0;1; 0;0;0;0;0;0;0;0;0;0;0;0; 0;1;1}; % number of hyperparameters of each token
dict_conds   = [0.01,0.05,0.1,0.15,0.2]; % candidate count conditions
dict_forks   = 1:seq_max_comp; % hyperparameters of fork pointer

% prepare
switch problem
    case 'pbo'
        [Problem,Data,Setting,seedTrain] = opt_env('Mode','design','Problem','pbo','InstanceTrain',instanceTrain,'InstanceTest',1,'Problem_id',problem_id,'eval',eval);
    case 'beamforming'
        [Problem,Data,Setting,seedTrain] = opt_env('Mode','design','Problem','beamforming','InstanceTrain',instanceTrain,'InstanceTest',3,'Problem_id',problem_id,'eval',eval);
    case 'blackstart'
        [Problem,Data,Setting,seedTrain] = opt_env('Mode','design','Problem','blackstart','InstanceTrain',instanceTrain,'InstanceTest',[],'Problem_id',problem_id,'eval',eval);
end
seq  = seq + 1; % python list from 0, add 1 in matlab
keys = [Setting.AllOp;other_token]; % all tokens
dict = [keys,values]; % for decoding the sequence representation of the designed algorithm

% decode the sequence
indOp     = []; % indexes of components
Paras     = cell(num_comp,2); % 17 components' hyperparameters
Conds     = zeros(num_comp,1); % 17 components, each has 1 condition
ParaSpace = Setting.ParaSpace;
indPtr    = []; % indexes of pointers
paraPtr   = []; % conditions associated with the iterate pointer or hyperparameter of the fork pointer
i = 1;
while i <= length(seq)
    thisToken = seq(i);
    numPara   = dict{thisToken,2};  % number of hyperparameters of this token

    if thisToken <= num_comp % if component
        indOp = [indOp,thisToken];
        if numPara > 0
            tempPara = zeros(numel(numPara),1);
            for j = 1:numel(numPara)
                para = dict{seq(i+j),1}; % the next token
                tempPara(j) = ParaSpace{thisToken}(:,1) + (ParaSpace{thisToken}(:,2)-ParaSpace{thisToken}(:,1))/9.*(para-1); % min+(max-min)/9*(para-1)
            end
            Paras{thisToken,1} = tempPara;
            i = i+1+numel(numPara);
        else
            i = i+1;
        end
    elseif thisToken > size(keys,1)-3 % if pointer
        indPtr  = [indPtr,thisToken];
        if numPara == 1 % iterate or fork pointer
            para = dict{seq(i+1),1}; 
            if thisToken == size(keys,1)-1 % if iterate pointer, para is the count condition
                para = dict_conds(para);
            elseif thisToken == size(keys,1) % if fork pointer, para is the hyperparameter of fork
                para = dict_forks(para) - 1; % dict_forks(para);
            end
            paraPtr = [paraPtr,para];
            i = i+2;
        elseif numPara == 0 % forward pointer
            paraPtr = [paraPtr,0];
            i = i+1;
        end
    end
end

indFork = find(indPtr==32); % indexes (in indOp) of components that are associated with the fork pointer
Setting.AlgP = numel(indFork)+1; % number of branches determined by the fork pointer
Operators = cell(1,Setting.AlgP);
for i = 1:Setting.AlgP
    % for the fork pointer and its hyperparameter
    if i == 1 % the first branch (whole sequence)
        indOpTemp = indOp;
    else % sequence segments determined by fork
        paraPtr(indFork(i-1)) = min(paraPtr(indFork(i-1)),length(indOp));
        indOpTemp = [indOp(1:indFork(i-1)),indOp(paraPtr(indFork(i-1)):end)];
    end
    
    opAdjMat = []; % operators' graph representation (adjacent matrix) to fit the AutoOpt software
    for j = 1:length(indOpTemp)-1
        opAdjMat = [opAdjMat;indOpTemp(1,j:j+1)]; 
    end
    Operators{i} = opAdjMat;
    
    % for the forward and iterate pointers and the count condition
    indSearch = indOpTemp(2:end-1); % indexes of search operators
    for k = 1:numel(indSearch)
        if indPtr(k+1) == size(keys,1)-1 % if the search component is associated with the iterate pointer
            Paras{indSearch(k),2} = 'LS'; % the search component iterate
            Conds(indSearch(k)) = paraPtr(k+1)*Setting.ProbFE;
        elseif indPtr(k+1) == size(keys,1)-2 % if the search component is associated with the forward pointer
            Paras{indSearch(k),2} = 'GS'; % the search component forward
        end
    end
end
Setting.Conds = Conds;

% evaluate the designed algorithm by the AutoOpt software
Alg = DESIGN;
Alg = Alg.Construct2(Operators,Paras,Problem,Setting);
[Alg,~]  = Alg.Evaluate(Problem,Data,Setting,seedTrain);
performance = Alg.performance;
end