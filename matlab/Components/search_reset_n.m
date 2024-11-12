function [output1,output2] = search_reset_n(varargin)
% Randomly select n elements of each solution and reset them to random
% values.

%------------------------------Copyright-----------------------------------
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

mode = varargin{end};
switch mode
    case 'execute'
        Solution = varargin{1};
        Problem  = varargin{2};
        Para     = varargin{3};
        Aux      = varargin{4};

        if ~isnumeric(Solution)
            New = Solution.decs;
        else
            New = Solution;
        end

        n = max(round(Para),1);
        [N,D] = size(New);

        for i = 1:N
            ind = randperm(D,n);
            curr = New(i,ind);
            for j = 1:n
                while New(i,ind(j)) == curr(j)
                    New(i,ind(j)) = randi([Problem.bound(1,ind(j)),Problem.bound(2,ind(j))]);
                end
            end
        end
        output1 = New;
        output2 = Aux;

    case 'parameter'
        % number of problem's decision variable
        Problem  = varargin{1};
        D = zeros(length(Problem),1);
        for i = 1:length(Problem)
            D(i) = size(Problem(i).bound,2);
        end
        D = min(D);
        n_max = D; % the maximum n
        output1 = [1,n_max];

    case 'behavior'
        output1 = {'LS','small';'GS','large'}; % small n values perform local search
end

if ~exist('output1','var')
    output1 = [];
end
if ~exist('output2','var')
    output2 = [];
end
end