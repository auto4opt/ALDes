% �ļ�����dijkstra.m
% ʱ�䣺2020��9��12��
% ��Դ��https://blog.csdn.net/lishan132/article/details/108527271
% ���ܣ�����dijkstra�㷨�������������·��
% dist��������յ�֮�����̾���ֵ
% path�����·������
% Distance�����·���µľ���ֵ
% A���ڽӾ���
% strat�������
% dest���յ���
function [dist,resultPath,Distance] = dijkstra(A,start,dest)
% �������� 
% �����������ʱ��
% tic  %��ʼ��ʱ

% ��ʼ������
p = size(A,1);        %���㶥����Ŀ 
S(1) = dest;          %��ʼ������S���Ѽ��뵽·���еĶ�����
U = 1:p;              %��ʼ������U��δ���뵽·���еĶ�����
U(dest) = [];         %ɾ���յ���
Distance = zeros(2,p);  %��ʼ�����ж��㵽�յ�dest�ľ���
Distance(1,:) = 1:p;    %�ظ�ֵ��һ��Ϊ��������
Distance(2,1:p) = A(dest,1:p);  %�ظ�ֵ�ڶ���Ϊ�ڽӾ����и����㵽�յ�ľ���
new_Distance = Distance;
D = Distance;            %��ʼ��U�����ж��㵽�յ�dest�ľ���
D(:,dest) = [];          %ɾ��U���յ��ŵ��յ��ŵľ���
path = zeros(2,p);  %��ʼ��·��
path(1,:) = 1:p;    %�ظ�ֵ��һ��Ϊ��������
path(2,Distance(2,:)~=inf) = dest;  %����ֵ��Ϊ�����ʱ��������������
 
% Ѱ�����·��
while ~isempty(U)  %�ж�U��Ԫ���Ƿ�Ϊ��
    index = find(D(2,:)==min(D(2,:)),1);  %ʣ�ඥ���о�����Сֵ������
    k = D(1,index);   %����ʣ�ඥ���о����յ�����Ķ�����
    
    %���¶���
    S = [S,k];     %������k��ӵ�S��
    U(U==k) = [];  %��U��ɾ������k  
    
    %�������
    new_Distance(2,:) = A(k,1:p)+Distance(2,k); %������ͨ�����k���ٴ�k�����յ�����е����ֵ
    D = min(Distance,new_Distance);  %��ԭ���ľ���ֵ�Ƚϣ�ȡ��Сֵ  
   
    %����·��
    path(2,D(2,:)~=Distance(2,:)) = k;  %�����µ���Сֵ���������ӹ�ϵ�����ӵ����k�� 
    
    %���¾���
    Distance = D;  %���¾����Ϊ���е㵽�յ����Сֵ
    D(:,S) = [];   %ɾ���Ѽ��뵽S�еĶ���
end
dist = Distance(2,start);  %ȡ��ָ����㵽�յ�ľ���ֵ
% toc %��ʱ����
 
% ������
% fprintf('�ҵ������·��Ϊ��');
resultPath = [];
while start ~= dest    %�����յ�ʱ����
%     fprintf('%d-->',start);  %��ӡ��ǰ����
    resultPath = [resultPath,start];
    next = path(2,start);    %�뵱ǰ����������һ����
    start = next;            %���µ�ǰ��
end
resultPath = [resultPath,start];
% fprintf('%d\n',dest);
% fprintf('���·����Ӧ�ľ���Ϊ��%d\n',dist);
end
 