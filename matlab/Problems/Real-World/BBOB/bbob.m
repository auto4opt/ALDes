function [output1,output2,output3] = bbob(varargin)
% The beanforming problem in RIS-aided communications.

%------------------------------Reference-----------------------------------
% Yan B, Zhao Q, Li M, et al. Fitness landscape analysis and niching 
% genetic approach for hybrid beamforming in RIS-aided communications[J]. 
% Applied Soft Computing, 2022, 131: 109725.
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

switch varargin{end}
    case 'construct' % define problem properties
        Problem  = varargin{1};
        instance = varargin{2};       
        Data     = struct('o',[]);
        for i = 1:length(instance)
            if instance(i) == 1
                D = 2;
            elseif instance(i) == 2
                D = 5;
            elseif instance(i) == 3
                D = 10;
            elseif instance(i) == 4
                D = 20;
            elseif instance(i) == 5
                D = 90;
            else
                error('Only instances 1, 2, and 3 are available.')
            end  
            
            lower = zeros(1,D)-5;
            upper = zeros(1,D)+5;
            
            Problem(i).type = {'continuous','static','certain'};
            Problem(i).bound = [lower;upper];
            %Problem(i).problem_index = 1
            Data(i).o = Data; % do nothing but add data(i)
        end
        
        output1 = Problem;
        output2 = Data;

    case 'repair' % repair solutions
        Decs = varargin{2};
        output1 = Decs;
    
    case 'evaluate' % evaluate solution's fitness
        problem_id  = varargin{3};
        Decs = varargin{2};
        [q,dim] = size(Decs);
        
        res = py.IOH_Test.BBOB(Decs,dim,problem_id);
        res = double(res).';
        output1 = res; % matrix for saving objective function values
end

if ~exist('output2','var')
    output2 = [];
end
if ~exist('output3','var')
    output3 = [];
end
end