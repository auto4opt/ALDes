function [Algs,AlgTrace]= Learn(Problem,Data,Setting,seedTrain,app,bar)
% Learn the algorithm distribution via transformer and reinforcement learning
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

str = 'Designing... ';
if ~isempty(app)
    app.TextArea.Value = str;
    drawnow;
else
    waitbar(100,bar,str);
end

% train transformer


Algs = getAlg(Action,Problem,Data,Setting,seedTrain); % get the algorithm from the action
p    = Algs.GetPerformance(Setting,seedTrain); % algorithm's performance


% output the final algorithm
Algs = getAlg(Action,Problem,Data,Setting,seedTrain);
AlgTrace = Algs;
end

function Algs = getAlg(Action,Problem,Data,Setting,seedTrain)
% get the algorithm from the action

AlgN = 1; % number of algorithms to be evaluated
obj  = DESIGN;

% encode the action as the graph representation of the algorithm
[Operators,Paras] = obj.Encode(Action,Setting,AlgN);

% repair the algorithm to keep it to be feasible
[Operators,Paras] = obj.Repair(Operators,Paras,Problem,Setting);

% construct the algorithm struct
Algs(1,AlgN) = DESIGN;
for i = 1:AlgN
    Algs(i).operator  = Operators(i,:);
    Algs(i).parameter = Paras{i};
    [currOp,currPara]   = Algs.Decode(Operators(i,:),Paras{i},Problem,Setting);
    Algs(i).operatorPheno  = currOp;
    Algs(i).parameterPheno = currPara;
    Algs(i).performance       = zeros(length(Problem),Setting.AlgRuns);
    Algs(i).performanceApprox = zeros(length(Problem),Setting.AlgRuns);
end

% evaluate the algorithm
Algs = Algs.Evaluate(Problem,Data,Setting,seedTrain);
end