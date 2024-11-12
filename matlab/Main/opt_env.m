function [Problem,Data,Setting,seedTrain] = opt_env(varargin)
% Set the AutoOpt software for algorithm evaluation

% get mode
if any(strcmp(varargin,'Mode'))
    Setting = struct;
    ind = find(strcmp(varargin,'Mode'));
    Setting.Mode = varargin{ind+1};
else
    error('Please set the mode to "design" or "solve".');
end

% get problem
if strcmp(Setting.Mode,'design')
    [prob,instanceTrain,instanceTest] = Input(varargin,Setting,'data');
elseif strcmp(Setting.Mode,'solve')
    [prob,~] = Input(varargin,Setting,'data');
end

% get problem id
if any(strcmp(varargin,'Problem_id'))
    ind = find(strcmp(varargin,'Problem_id'));
    Setting.Problem_id = varargin{ind+1};
end

if any(strcmp(varargin,'eval'))
    ind = find(strcmp(varargin,'eval'));
    Setting.eval = varargin{ind+1};
end

% default parameters
switch Setting.Mode
    case 'design'
        Setting.AlgP     = 1;
        Setting.AlgQ     = 3;
        Setting.Archive  = '';
        Setting.IncRate  = -inf;
        Setting.ProbN    = 50;
        Setting.ProbFE   = 10000; %test set 50000  train set 5000,3000,10000
        Setting.InnerFE  = 200;
        Setting.AlgN     = 5;
        Setting.AlgFE    = 15000;
        Setting.AlgRuns  = 5; %test set 30 train set 5
        if Setting.eval== 1
            Setting.ProbFE   = 50000; %test set 50000  train set 5000
            Setting.AlgRuns  = 30; %test set 30 train set 5 % 1 for get convergence curve
        end
        Setting.Metric   = 'quality'; % quality/runtimeFE/runtimeSec/auc
        Setting.Generate = 'learn';   % search/learn
        Setting.Evaluate = 'exact';   % exact/approximate/intensification/racing
        Setting.Compare  = 'average'; % average/statistic
        Setting.Tmax     = [];
        Setting.Thres    = [];
        Setting.LSRange  = 0.3;
        Setting.RacingK  = max(1,round(length(instanceTrain)*0.2));
        Setting.Surro    = Setting.ProbFE*0.3;
        Setting          = Input(varargin,Setting,'parameter'); % replace default parameters with user-defined ones
        Setting          = Input(Setting,'check'); % avoid conflicting parameter settings

        %% construct training problem properties
        Problem  = struct('name',[],'type',[],'bound',[],'setting',{''},'N',[],'Gmax',[]);
        seedTrain     = randperm(numel(instanceTrain));
        seedTest      = randperm(numel(instanceTest))+length(seedTrain);
        instance      = [instanceTrain,instanceTest];

        for i = 1:numel(instance)
            Problem(i).name    = prob;
            Problem(i).setting = '';
            Problem(i).N       = Setting.ProbN;
            Problem(i).Gmax    = ceil(Setting.ProbFE/Setting.ProbN)-1;
            Problem(i).problem_id = Setting.Problem_id;
        end
        [Problem,Data,~] = feval(str2func(Problem(1).name),Problem,instance,'construct'); % infill problems' constraints and search boundary, construct data properties

        %% design algorithms
        Setting = Space(Problem,Setting); % get design space
end

