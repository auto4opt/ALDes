% 文件名：dijkstra.m
% 时间：2020年9月12日
% 来源：https://blog.csdn.net/lishan132/article/details/108527271
% 功能：利用dijkstra算法计算两点间的最短路径
% dist：起点与终点之间的最短距离值
% path：最短路径索引
% Distance：最短路径下的距离值
% A：邻接矩阵
% strat：起点编号
% dest：终点编号
function [dist,resultPath,Distance] = dijkstra(A,start,dest)
% 测试数据 
% 计算程序运行时间
% tic  %开始计时

% 初始化操作
p = size(A,1);        %计算顶点数目 
S(1) = dest;          %初始化集合S，已加入到路径中的顶点编号
U = 1:p;              %初始化集合U，未加入到路径中的顶点编号
U(dest) = [];         %删除终点编号
Distance = zeros(2,p);  %初始化所有顶点到终点dest的距离
Distance(1,:) = 1:p;    %重赋值第一行为各顶点编号
Distance(2,1:p) = A(dest,1:p);  %重赋值第二行为邻接矩阵中各顶点到终点的距离
new_Distance = Distance;
D = Distance;            %初始化U中所有顶点到终点dest的距离
D(:,dest) = [];          %删除U中终点编号到终点编号的距离
path = zeros(2,p);  %初始化路径
path(1,:) = 1:p;    %重赋值第一行为各顶点编号
path(2,Distance(2,:)~=inf) = dest;  %距离值不为无穷大时，将两顶点相连
 
% 寻找最短路径
while ~isempty(U)  %判断U中元素是否为空
    index = find(D(2,:)==min(D(2,:)),1);  %剩余顶点中距离最小值的索引
    k = D(1,index);   %发现剩余顶点中距离终点最近的顶点编号
    
    %更新顶点
    S = [S,k];     %将顶点k添加到S中
    U(U==k) = [];  %从U中删除顶点k  
    
    %计算距离
    new_Distance(2,:) = A(k,1:p)+Distance(2,k); %计算先通过结点k，再从k到达终点的所有点距离值
    D = min(Distance,new_Distance);  %与原来的距离值比较，取最小值  
   
    %更新路径
    path(2,D(2,:)~=Distance(2,:)) = k;  %出现新的最小值，更改连接关系，连接到结点k上 
    
    %更新距离
    Distance = D;  %更新距离表为所有点到终点的最小值
    D(:,S) = [];   %删除已加入到S中的顶点
end
dist = Distance(2,start);  %取出指定起点到终点的距离值
% toc %计时结束
 
% 输出结果
% fprintf('找到的最短路径为：');
resultPath = [];
while start ~= dest    %到达终点时结束
%     fprintf('%d-->',start);  %打印当前点编号
    resultPath = [resultPath,start];
    next = path(2,start);    %与当前点相连的下一顶点
    start = next;            %更新当前点
end
resultPath = [resultPath,start];
% fprintf('%d\n',dest);
% fprintf('最短路径对应的距离为：%d\n',dist);
end
 