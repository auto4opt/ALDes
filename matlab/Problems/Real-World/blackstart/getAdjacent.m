function [A,indBS,indNBS,indLoad] = getAdjacent(k)
%function [A,node] = getAdjacent
A = readmatrix('adjacent.xlsx','Sheet',1);
A(1,:) = []; % delete indexes
A(:,1) = []; % delete indexes
for i = 1:size(A,1)
    for j = 1:size(A,2)
        if ~isnan(A(i,j))
            A(j,i) = A(i,j);
        else
            A(i,j) = inf; % replace Nan with Inf
        end
    end
end

% A = [0,275.5,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,167.6;%1
%     275.5,0,101.2,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,57.6,inf,inf,inf,10,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf;%2
%     inf,101.2,0,142.8,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,89.1,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf;%3
%     inf,inf,142.8,0,85.8,inf,inf,inf,inf,inf,inf,inf,inf,86.5,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf;%4
%     inf,inf,inf,85.8,0,17.4,inf,75.1,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf;%5
%     inf,inf,inf,inf,17.4,0,61.7,inf,inf,inf,55,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,10,inf,inf,inf,inf,inf,inf,inf,inf;%6
%     inf,inf,inf,inf,inf,61.7,0,30.8,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf;%7
%     inf,inf,inf,inf,75.1,inf,30.8,0,243.3,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf;%8
%     inf,inf,inf,inf,inf,inf,inf,243.3,0,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,167.6;%9
%     inf,inf,inf,inf,inf,inf,inf,inf,inf,0,28.8,inf,28.8,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,10,inf,inf,inf,inf,inf,inf,inf;%10
%     inf,inf,inf,inf,inf,55.0,inf,inf,inf,28.8,0,10,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf;%11
%     inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,10,0,10,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf;%12
%     inf,inf,inf,inf,inf,inf,inf,inf,inf,28.8,inf,10,0,67.7,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf;%13
%     inf,inf,inf,86.5,inf,inf,inf,inf,inf,inf,inf,inf,67.7,0,145.4,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf;%14
%     inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,145.4,0,63,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf;%15
%     inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,63,0,59.7,inf,130.7,inf,90.5,inf,inf,39.5,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf;%16
%     inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,59.7,0,55,inf,inf,inf,inf,inf,inf,inf,inf,116,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf;%17
%     inf,inf,89.1,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,55,0,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf;%18
%     inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,130.7,inf,inf,0,10,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,10,inf,inf,inf,inf,inf,inf;%19
%     inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,10,0,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,10,inf,inf,inf,inf,inf;%20
%     inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,90.5,inf,inf,inf,inf,0,93.8,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf;%21
%     inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,93.8,0,64.3,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,10,inf,inf,inf,inf;%22
%     inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,64.3,0,234.6,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,10,inf,inf,inf;%23
%     inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,39.5,inf,inf,inf,inf,inf,inf,234.6,0,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf;%24
%     inf,57.6,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,0,216.5,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,10,inf,inf;%25
%     inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,216.5,0,98.5,317.7,418.9,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf;%26
%     inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,116,inf,inf,inf,inf,inf,inf,inf,inf,98.5,0,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf;%27
%     inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,317.7,inf,0,101.2,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf;%28
%     inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,418.9,inf,101.2,0,inf,inf,inf,inf,inf,inf,inf,inf,10,inf;%29
%     inf,10,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,0,inf,inf,inf,inf,inf,inf,inf,inf,inf;%30
%     inf,inf,inf,inf,inf,10,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,0,inf,inf,inf,inf,inf,inf,inf,inf;%31
%     inf,inf,inf,inf,inf,inf,inf,inf,inf,10,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,0,inf,inf,inf,inf,inf,inf,inf;%32
%     inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,10,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,0,inf,inf,inf,inf,inf,inf;%33
%     inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,10,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,0,inf,inf,inf,inf,inf;%34
%     inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,10,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,0,inf,inf,inf,inf;%35
%     inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,10,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,0,inf,inf,inf;%36
%     inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,10,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,0,inf,inf;%37
%     inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,10,inf,inf,inf,inf,inf,inf,inf,inf,0,inf;%38
%     167.6,inf,inf,inf,inf,inf,inf,inf,167.6,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,0];%39
if k == 1 % instance 1
    indBS = [30,31]; 
    indNBS = 32:39;
elseif k == 2
    indBS = [32,36];
    indNBS = [30:31,33:35,37:39];
end
indLoad = [1,3,4,7,8,9,12,15,16,18,20,21,23:29,31,39];
end