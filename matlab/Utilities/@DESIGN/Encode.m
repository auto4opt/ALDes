function [Operators,Paras] = Encode(Action,Setting,AlgN)
% Encode the action to the graph representation of the algorithm.
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

OpSpace   = Setting.OpSpace;
ParaSpace = Setting.ParaSpace;
Operators = cell(AlgN,Setting.AlgP); 
Paras     = cell(AlgN,1);
Action    = reshape(Action,3,numel(Action)/3)'; % one row for one operator

numChoose = OpSpace(1,2)-OpSpace(1,1)+1;
numSearch = OpSpace(2,2)-OpSpace(2,1)+1;
numUpdate = OpSpace(3,2)-OpSpace(3,1)+1;

for i = 1:AlgN
    %% encode operators to graph edges
    for j = 1:Setting.AlgP
        Operators{i,j} = zeros(Setting.AlgQ+1,2);
        
        % get operators' indexes
        indChoose = ceil(Action(1,1)*numChoose);       
        indSearch = zeros(size(Action,1)-2,1);
        for k = 1:length(indSearch)
            indSearch(k) = ceil(Action(k+2,1)*numSearch)+numChoose;
        end
        indUpdate = ceil(Action(2,1)*numUpdate)+numChoose+numSearch;

        % encode indexes to graph edges
        Operators{i,j}(1,:) = [indChoose,indSearch(1)];
        indSearchStart = indSearch(1);
        for k = 2:numel(indSearch)
            indSearchEnd = indSearch(k);
            Operators{i,j}(k,:) = [indSearchStart,indSearchEnd];
            indSearchStart = indSearchEnd;
        end
        Operators{i,j}(end,:) = [indSearch(end),indUpdate];
    end

    %% encode parameters
    % initialize all operators' parameters
    tempPara = cell(length(ParaSpace),2);
    indNonEmptyPara = find(~cellfun(@isempty,ParaSpace)==1);
    for j = 1:numel(indNonEmptyPara)
        k = indNonEmptyPara(j);
        tempPara{k,1} = ParaSpace{k}(:,1)+(ParaSpace{k}(:,2)-ParaSpace{k}(:,1)).*rand(size(ParaSpace{k},1),1); 
    end
    
    % replease the involved operators' parameters with ones given by the action
    j = 1; choosePara = [];
    while j <= size(ParaSpace{indChoose},1) % for each parameter
        choosePara = [choosePara;Action(1,1+j)];
        j = j+1;
    end
    if ~isempty(choosePara)
        tempPara{indChoose,1} = ParaSpace{indChoose}(:,1)+choosePara.*(ParaSpace{indChoose}(:,2)-ParaSpace{indChoose}(:,1));
    end

    for j = 1:numel(indSearch)
        k = 1; searchPara = [];
        while k <= size(ParaSpace{indSearch(j)},1)
            searchPara = [searchPara;Action(2+j,1+k)];
            k = k+1;
        end
        if ~isempty(searchPara)
            tempPara{indSearch(j),1} = ParaSpace{indSearch(j)}(:,1)+searchPara.*(ParaSpace{indSearch(j)}(:,2)-ParaSpace{indSearch(j)}(:,1));
        end
    end

    j = 1; updatePara = [];
    while j <= size(ParaSpace{indUpdate},1) 
        updatePara = [updatePara;Action(2,1+j)];
        j = j+1;
    end
    if ~isempty(updatePara)
        tempPara{indUpdate,1} = ParaSpace{indUpdate}(:,1)+updatePara.*(ParaSpace{indUpdate}(:,2)-ParaSpace{indUpdate}(:,1));
    end

    Paras{i} = tempPara;
end
end