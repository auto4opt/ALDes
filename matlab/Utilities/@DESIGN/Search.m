function [Algs,AlgTrace]= Search(Problem,Data,Setting,seedTrain,app,bar)
% Search algorithms via heuristics.
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

% initialize algorithms and evaluate their performance
AlgGmax = ceil(Setting.AlgFE/Setting.AlgN);
switch Setting.Evaluate
    case {'exact','intensification'}
        Algs = DESIGN(Problem,Setting);
        Algs = Algs.Evaluate(Problem,Data,Setting,seedTrain);
    case 'racing'
        Algs = DESIGN(Problem,Setting);
        Algs = Algs.Evaluate(Problem,Data,Setting,seedTrain(1:Setting.RacingK));
    case 'approximate'
        Surrogate = Approximate(Problem,Data,Setting,seedTrain); % initialize surrogate model
        Algs = Surrogate.data(randperm(length(Surrogate.data),Setting.AlgN)); % get initial algorithms
end

% iterate search
G = 1;
AlgTrace = DESIGN; % for save best algorithms found at each iteration
while G <= AlgGmax
    str = ['Designing... ',num2str(100*G/AlgGmax),'%'];
    if ~isempty(app)
        app.TextArea.Value = str;
        drawnow;
    else
        waitbar(G/AlgGmax,bar,str);
    end
    improve = 1;
    innerG  = 1;
    Aux     = cell(Setting.AlgN,1); % auxiliary data
    if Setting.AlgN == 1
        innerGmax = ceil(Setting.AlgFE/Setting.AlgN/10);
    else % global search for designing multiple algorithms
        innerGmax = 1;
    end
    while improve(1) >= Setting.IncRate && innerG <= innerGmax
        % design new algorithms
        [NewAlgs,Aux] = Algs.GetNew(Problem,Setting,innerG,Aux);

        % performance evaluation and algorithm selection
        switch Setting.Evaluate
            case 'exact'
                NewAlgs = NewAlgs.Evaluate(Problem,Data,Setting,seedTrain);
                AllAlgs = [Algs,NewAlgs];
                Algs = AllAlgs.Select(Problem,Data,Setting,seedTrain);

            case 'approximate'
                % get surrogate
                NewAlgs = NewAlgs.Estimate(Problem,Setting,seedTrain,Surrogate);
                AllAlgs = [Algs,NewAlgs];
                Algs = AllAlgs.Select(Problem,Data,Setting,seedTrain);
                % update surrogate
                if ismember(G,Surrogate.exactG)
                    NewAlgs = NewAlgs.Evaluate(Problem,Data,Setting,seedTrain);
                    Surrogate = Surrogate.UpdateModel(NewAlgs,Setting);
                end

            case 'intensification'
                % screen survivals from new algorithms
                while ~isempty(NewAlgs) && ~isempty(seedTrain)
                    NewAlgs = NewAlgs.Evaluate(Problem,Data,Setting,seedTrain(1));
                    AllAlgs = [Algs,NewAlgs];
                    NewAlgs = AllAlgs.Select(Problem,Data,Setting,seedTrain(1));
                    seedTrain(1) = [];
                end
                % restore instance indices
                seedTrain = randperm(numel(instance));
                % evaluate new incumbents (NewAlgs)' performance on all instances
                for i = 1:length(NewAlgs)
                    for j = 1:numel(seedTrain)
                        if sum(NewAlgs(i).performance(seedTrain(j),:)) == 0 % if haven't evaluated on instance j
                            NewAlgs(i) = NewAlgs(i).Evaluate(Problem,Data,Setting,seedTrain(j));
                        end
                    end
                end
                % update incumbent algorithms
                Algs(randperm(Setting.AlgN,length(NewAlgs))) = NewAlgs;

            case 'racing'
                % screen survivals from all algorithms (racing)
                NewAlgs = NewAlgs.Evaluate(Problem,Data,Setting,seedTrain(1:Setting.RacingK));
                AllAlgs = [Algs,NewAlgs];
                Algs = AllAlgs.Select(Problem,Data,Setting,seedTrain(1:Setting.RacingK));
                seedTrain(1:Setting.RacingK) = [];
                while length(Algs) > Setting.AlgN && ~isempty(seedTrain)
                    Algs = Algs.Select(Problem,Data,Setting,seedTrain(1));
                    seedTrain(1) = [];
                end
                % restore instance indices
                seedTrain = randperm(numel(instance));
                % delete redundant algorithms after racing
                if length(Algs) > Setting.AlgN
                    ind = randperm(length(Algs),length(Algs)-Setting.AlgN);
                    Algs(ind) = [];
                end
        end

        % update auxiliary data
        for i = 1:Setting.AlgN
            if isfield(Aux{i},'cma_Disturb') % if use CMA-ES
                Aux{i} = para_cmaes(Algs(i),Problem,Aux{i},'algorithm'); % update CMA-ES's parameters
            end
        end

        % record best algorithms at each iteration
        currCompare = Setting.Compare;
        Setting.Compare = 'average';
        currPerform = Algs.GetPerformance(Setting,seedTrain);
        Setting.Compare = currCompare;
        [~,best] = min(currPerform);
        AlgTrace(G) = Algs(best);

        improve = ImproveRate(Algs,improve,innerG,'algorithm');
        innerG  = innerG+1;
        G       = G+1;
        if G > AlgGmax
            break
        end
    end
end