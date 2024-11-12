function [output1,output2] = Process(varargin)
% The main process of algorithm design and problem solving.

%----------------------------Copyright-------------------------------------
% Copyright (C) <2023>  <Swarm Intelligence Lab>

% AutoOptLib is a free software. You can use, redistribute, and/or modify
% it under the terms of the GNU General Public License as published by the
% Free Software Foundation, either version 3 of the License, or any later
% version.

% Please reference the paper below if using AutoOptLib in your publication:
% @article{zhao2023autooptlib,
%  title={AutoOptLib: A Library of Automatically Designing Metaheuristic
%         Optimization Algorithms in Matlab},
%  author={Zhao, Qi and Yan, Bai and Hu, Taiwei and Chen, Xianglong and
%          Yang, Jian and Shi, Yuhui},
%  journal={arXiv preprint 	arXiv:2303.06536},
%  year={2023}
% }
%--------------------------------------------------------------------------

if strcmp(varargin{1}(end:-1:end-1),'m.')
    varargin{1}(end:-1:end-1)=[]; % delete '.m'
end
Setting = varargin{end};
switch Setting.Mode
    case 'design'
        tic;
        str = 'Initializing...';
        if nargin > 4
            app = varargin{end-1};
            app.TextArea.Value = str;
            drawnow;
            bar = [];
        else
            app = [];
            bar = waitbar(0,str);
        end
        %% construct training problem properties
        Problem  = struct('name',[],'type',[],'bound',[],'setting',{''},'N',[],'Gmax',[]);
        instanceTrain = varargin{2};
        instanceTest  = varargin{3};
        seedTrain     = randperm(numel(instanceTrain));
        seedTest      = randperm(numel(instanceTest))+length(seedTrain);
        instance      = [instanceTrain,instanceTest];
        for i = 1:numel(instance)
            Problem(i).name    = varargin{1};
            Problem(i).setting = '';
            Problem(i).N       = Setting.ProbN;
            Problem(i).Gmax    = ceil(Setting.ProbFE/Setting.ProbN);
            Problem(i).problem_id = Setting.problem_id;
        end
        [Problem,Data,~] = feval(str2func(Problem(1).name),Problem,instance,'construct'); % infill problems' constraints and search boundary, construct data properties


        %% design algorithms
        Setting = Space(Problem,Setting); % get design space
        obj = DESIGN;
        switch Setting.Generate
            case 'search' % search algorithms via heuristics
                [Algs,AlgTrace] = obj.Search(Problem,Data,Setting,seedTrain,app,bar);       

            case 'learn' % learn the algorithm distribution via transformer and reinforcement learning
                [Algs,AlgTrace] = obj.Learn(Problem,Data,Setting,seedTrain,app,bar);
            case 'get-p'
                output1 = obj.Get_performance(Problem,Data,Setting,seedTrain);
                return;
        end

        %% test the designed algorithms
        str = 'Testing... ';
        if ~isempty(app)
            app.TextArea.Value = str;
            drawnow;
        else
            waitbar(100,bar,str);
        end
        Setting.Evaluate = 'exact';
        Algs = Algs.Evaluate(Problem,Data,Setting,seedTest);
        Algs = Algs.Select(Problem,Data,Setting,seedTest); % sort algorithms in descending order in terms of their performance
        output1 = Algs; % final algorithms
        output2 = AlgTrace; % best algorithms found at each iteration of design

        str = 'Complete';
        if ~isempty(app)
            app.TextArea.Value = str;
            drawnow;
        else
            waitbar(100,bar,str);
        end
        toc;

    case 'solve'
        %% construct problem properties
        ProblemSolve  = struct('name',[],'type',[],'bound',[],'setting',{''},'N',[],'Gmax',[]);
        instanceSolve = varargin{2};
        for i = 1:numel(instanceSolve)
            ProblemSolve(i).name = varargin{1};
            ProblemSolve(i).setting = '';
            ProblemSolve(i).N    = Setting.ProbN;
            ProblemSolve(i).Gmax = ceil(Setting.ProbFE/Setting.ProbN);
            ProblemSolve(i).problem_id =Setting.problem_id;    
        end
        [ProblemSolve,DataSolve,~] = feval(str2func(ProblemSolve(1).name),ProblemSolve,instanceSolve,'construct'); % infill problems' constraints and search boundary, construct data properties

        %% solve the problem
        Solution = SOLVE;
        [Alg,Setting] = Solution.InputAlg(Setting); % algorithm profile
        if nargin > 3
            app = varargin{end-1};
            [bestSolutions,allSolutions] = Solution.RunAlg(Alg,ProblemSolve,DataSolve,app,Setting);
        else
            [bestSolutions,allSolutions] = Solution.RunAlg(Alg,ProblemSolve,DataSolve,[],Setting);
        end
        output1 = bestSolutions; % the best solution at the final iteration of each algorithm run
        output2 = allSolutions;  % the best solution at each iteration of the best algorithm run
end
end