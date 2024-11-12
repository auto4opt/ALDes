function [output1,output2,output3] = blackstart(varargin)
% The black start problem

switch varargin{end}
    case 'construct'
        Problem  = varargin{1};
        instance = varargin{2};
        % define the bound of decision space
        a = 2;  % number of BS
        b = 8;  % number of NBS
        c = 21; % number of load
        lower = zeros(1,a*b+b*c); % 1*D, lower bound of the D-dimension decision space
        upper = ones(1,a*b+b*c);  % 1*D, upper bound of the D-dimension decision space
        for i = 1:length(instance)
            Problem(i) = Problem(1);
            Problem(i).bound = [lower;upper];

            % define problem type in the following three cells (optional), type
            % can be defined either here or in the Main file
            % choices:
            % first cell : 'continuous'\'discrete'\'permutation'
            % second cell: 'static'\'sequential'
            % third cell : 'certain'\'uncertain'
            Problem(i).type = {'discrete','static','certain'};
        end
        output1 = Problem;

        % load data file
        orgData = case3901;
        Weight  = readmatrix('weight.xlsx','Sheet',1); % the weight of the Loads
        Data = struct;
        for k = 1:length(instance)
            if k == 1
                BS  = orgData.gen([1,2],:);
                NBS = orgData.gen(3:10,:);
            elseif k == 2
                BS  = orgData.gen([3,7],:);
                NBS = orgData.gen([1:2,4:6,8:10],:);
            end
            Load    = orgData.bus([1,3,4,7,8,9,12,15,16,18,20,21,23:29,31,39],:);
            branch  = orgData.branch;
            bus     = orgData.bus;

            % calculate the distance between any two nodes
            [A,indBS,indNBS,indLoad] = getAdjacent(k);
            w = zeros(numel(indBS),numel(indNBS));   % for save distance between BS and NBS
            l = zeros(numel(indNBS),numel(indLoad)); % for save distance between NBS and Load
            BSNBSPath = cell(numel(indBS),numel(indNBS)); % for save path from BS to NBS
            NBSLoadPath = cell(numel(indNBS),numel(indLoad)); % for save path from NBS to Load
            for i = 1:numel(indBS)
                for j = 1:numel(indNBS)
                    start = indBS(i);
                    dest = indNBS(j);
                    [w(i,j),BSNBSPath{i,j},~] = dijkstra(A,start,dest);
                end
            end
            for i = 1:numel(indNBS)
                for j = 1:numel(indLoad)
                    start = indNBS(i);
                    dest = indLoad(j);
                    [l(i,j),NBSLoadPath{i,j},~ ] = dijkstra(A,start,dest);
                end
            end

            Data(k).BS = BS;
            Data(k).NBS = NBS;
            Data(k).Load = Load;
            Data(k).branch = branch;
            Data(k).bus = bus;
            Data(k).w = w;
            Data(k).l = l;
            Data(k).BSNBSPath = BSNBSPath;
            Data(k).NBSLoadPath = NBSLoadPath;
            Data(k).Weight = Weight;
        end
        output2 = Data;

    case 'repair'
        Data = varargin{1};
        Decs = varargin{2};
        a = size(Data.BS,1); % number of BS
        b = size(Data.NBS,1); % number of NBS
        c = size(Data.Load,1); % number of load
        for i = 1:size(Decs,1)
            v = reshape(Decs(i,1:a*b),b,a);
            v = v'; % 2*8(a*b)
            h = reshape(Decs(i,a*b+1:end),c,b);
            h = h'; % 8*21(b*c)
            %repair the solution
            % only one BS to each NBS
            for g = 1:b % for each NBS
                if sum(v(:,g)) > 1
                    BSInd = find(v(:,g)==1);
                    randInd = randperm(numel(BSInd));
                    v(BSInd(randInd(1:end-1)),g) = 0;
                elseif sum(v(:,g)) < 1
                    randInd = randi(a);
                    v(randInd,g) = 1;
                end
            end
            % for each load,only one NBS to each load
            for k = 1:c
                if sum(h(:,k)) > 1
                    NBSInd = find(h(:,k)==1);
                    randInd = randperm(numel(NBSInd));
                    h(NBSInd(randInd(1:end-1)),k) = 0;
                elseif sum(h(:,k)) < 1
                    randInd = randi(b);
                    h(randInd,k) = 1;
                end
            end
            %give v&h to the Decs(i)
            f = v';
            f = reshape(f,1,a*b);
            e = h';
            e = reshape(e,1,b*c);
            Decs(i,1:a*b) = f;
            Decs(i,a*b+1:end) = e;
        end
        output1 = Decs;

    case 'evaluate'
        Data = varargin{1}; % load problem data
        Decs = varargin{2}; % load the current solution(s)

        % define the objective function in the following
        a = size(Data.BS,1); % number of BS
        b = size(Data.NBS,1); % number of NBS
        c = size(Data.Load,1); % number of load
        w = Data.w; % distance between BS and NBS
        l = Data.l;% distance between NBS and Load
        Objs = zeros(size(Decs,1),1);
        for i = 1:size(Decs,1)
            v = reshape(Decs(i,1:a*b),b,a);
            v = v'; % 2*8(a*b)
            h = reshape(Decs(i,a*b+1:end),c,b);
            h = h'; % 8*21(b*c)
            F = zeros(b,1);
            for j = 1:b
                F(j) = v(:,j)'*w(:,j);
            end
            F = sum(F);
            G = zeros(c,1);
            for k = 1:c
                G(k) = h(:,k)'*l(:,k);
            end
            G = sum(G);
            Objs(i) = F+G;
        end


        % calculate the priority of NBS
        Weight = Data.Weight;
        maxW = zeros(b,1); % power of the most important load of each NBS
        for i = 1:size(h,1)
            indLoadThisNBS = find(h(i,:)==1); % indexes of the loads belong to NBS i
            indLoadThisNBS = indLoadThisNBS';
            if isempty(indLoadThisNBS) == 0
                WeightThisNBS = Weight(indLoadThisNBS,1);
            else
                WeightThisNBS = 0;
            end
            maxW(i) = max(WeightThisNBS);
        end
        [~,priority] = sort(maxW,'descend'); % run NBS (index) one by one according to priority

        % define the inequal constraint(s) in the following, equal
        % constraints should be transformed to inequal ones
        restoredbranchInd = {};
        CV = zeros(size(Decs,1),1);

        for i = 1:size(Decs,1)
            % extract the path according to solution
            BSNBSInd = zeros(1,b);  %哪个BS负责哪个NBS（对应的位置）
            BSNBSPath = Data.BSNBSPath;%BS to NBS的路径
            thisBSNBSPath = cell(b,1);%根据解，提取的BStoNBS的路径
            NBSLoadInd = zeros(1,c);%哪个NBS负责哪个Load（对应的位置）
            NBSLoadPath = Data.NBSLoadPath;%NBS to Load的路径
            thisNBSLoadPath = cell(c,1);%根据解，提取的NBStoLoad的路径
            % 根据solution，提取BS到NBS的路径
            for j = 1:size(v,2)
                BSNBSInd(j) = find(v(:,j)==1);
                thisBSNBSPath{j} =  BSNBSPath{BSNBSInd(j),j};
            end
            % 根据solution，提取NBS到Load的路径
            for j = 1:size(h,2)
                NBSLoadInd(j) = find(h(:,j)==1);
                thisNBSLoadPath{j} =  NBSLoadPath{NBSLoadInd(j),j};
            end

            %更新各部门状态
            ubranch = zeros(size(Data.branch,1),1);% branch的状态
            ust = zeros(b,1); % starting state of NBS
            uramp = zeros(b,1); % rumping state of NBS
            %             ubus = zeros(size(Data.bus,1),1);
            %             ubus(30:31) = 1;
            %BS恢复，then与BS直接相连的line恢复(与BS节点号一样的line)
            for j = 1:size(Data.BS,1)
                thisBSbranchind = Data.branch(:,2)==Data.BS(j,1);%找到与BS节点号相等的tbus的位置（这个位置也就是branch的编号）
                ubranch(thisBSbranchind) = 1;
            end

            %对line，邻线恢复，才能恢复(两条线有共同的节点)
            %根据NBS优先级，检索BStoNBS的each路径，更新状态
            for m = 1:numel(priority)
                for k = 1:numel(thisBSNBSPath{priority(m)}(1:end-1))
                    for j = 1:size(Data.branch,1)
                        if ubranch(j) == 1
                            if thisBSNBSPath{priority(m)}(k) == Data.branch(j,1)
                                connBusInd = k;%在thisBSNBSPath中找到与branchj有共同节点的节点号的位置（在thisBSNBSPath的priority（m）行的第几个）
                                if thisBSNBSPath{priority(m)}(connBusInd)<thisBSNBSPath{priority(m)}(connBusInd+1)%根据节点号，找到thisBSNBSPath中次节点号与下一节点号组成的branch的ind
                                    allconnBranchInd = find(Data.branch(:,1)==thisBSNBSPath{priority(m)}(connBusInd));%根据节点号，在branch的起点中找到与该节点号相同的所有branch的编号
                                    connBranchInd = find(Data.branch(allconnBranchInd,2)==thisBSNBSPath{priority(m)}(connBusInd+1));%在找到的编号里，在branch的终点中找出与下一个节点号相同的branch的编号
                                    %connBranchInd = find(Data.branch(:,1)==thisBSNBSPath{priority(m)}(connBusInd)&&Data.branch(:,2)==thisBSNBSPath{priority(m)}(connBusInd+1));%找到的节点号比排在他后面的节点号小
                                else
                                    allconnBranchInd = find(Data.branch(:,1)==thisBSNBSPath{priority(m)}(connBusInd+1));
                                    connBranchInd = find(Data.branch(allconnBranchInd,2)==thisBSNBSPath{priority(m)}(connBusInd));
                                    %connBranchInd = find(Data.branch(:,1)==thisBSNBSPath{priority(m)}(connBusInd+1)&&Data.branch(:,2)==thisBSNBSPath{priority(m)}(connBusInd));%找到的节点号比排在他后面的节点号大
                                end
                                ubranch(allconnBranchInd(connBranchInd))=1;
                                restoredbranchInd = {restoredbranchInd,allconnBranchInd(connBranchInd)};
                            end
                            if thisBSNBSPath{priority(m)}(k) == Data.branch(j,2)
                                connBusInd = k;%在thisBSNBSPath中找到与branchj有共同节点的节点号的位置（在thisBSNBSPath的priority（m）行的第几个）
                                if thisBSNBSPath{priority(m)}(connBusInd)<thisBSNBSPath{priority(m)}(connBusInd+1)%根据节点号，找到thisBSNBSPath中次节点号与下一节点号组成的branch的ind
                                    allconnBranchInd = find(Data.branch(:,1)==thisBSNBSPath{priority(m)}(connBusInd));
                                    connBranchInd = find(Data.branch(allconnBranchInd,2)==thisBSNBSPath{priority(m)}(connBusInd+1));
                                    %connBranchInd = find(Data.branch(:,1)==thisBSNBSPath{priority(m)}(connBusInd)&&Data.branch(:,2)==thisBSNBSPath{priority(m)}(connBusInd+1));%找到的节点号比排在他后面的节点号小
                                else
                                    allconnBranchInd = find(Data.branch(:,1)==thisBSNBSPath{priority(m)}(connBusInd+1));
                                    connBranchInd = find(Data.branch(allconnBranchInd,2)==thisBSNBSPath{priority(m)}(connBusInd));
                                    %connBranchInd = find(Data.branch(:,1)==thisBSNBSPath{priority(m)}(connBusInd+1)&&Data.branch(:,2)==thisBSNBSPath{priority(m)}(connBusInd));%找到的节点号比排在他后面的节点号大
                                end
                                ubranch(allconnBranchInd(connBranchInd))=1;
                                restoredbranchInd = {restoredbranchInd,allconnBranchInd(connBranchInd)};
                            end
                        end
                    end
                end
            end

            %更新NBStoLoad的状态
            for m = 1:size(thisNBSLoadPath,1)
                for k = 1:numel(thisNBSLoadPath{m}(1:end-1))
                    for j = 1:size(Data.branch,1)
                        if ubranch(j) == 1
                            if thisNBSLoadPath{m}(k) == Data.branch(j,1)%路径m的第k个与branchj起点相同
                                connBusInd = k;%在thisNBSLoadPath里，找到与branchj的起点或终点相同的那个节点号所在的位置
                                if thisNBSLoadPath{m}(connBusInd)<thisNBSLoadPath{m}(connBusInd+1)%如果找到的节点号在thisNBSLoadPath里对应的节点号比下一个小
                                    allconnBranchInd = find(Data.branch(:,1)== thisNBSLoadPath{m}(connBusInd));%在branch中，找到所有起点与该节点相同的branch编号
                                    connBranchInd = find(Data.branch(allconnBranchInd,2)==thisNBSLoadPath{m}(connBusInd+1));%在上述branch编号中，找到终点与该节点下一个节点相同的branch编号（唯一）
                                    %connBranchInd = find(Data.branch(:,1)== thisNBSLoadPath{m}(connBusInd)&&Data.branch(:,2)==thisNBSLoadPath{m}(connBusInd+1));%起点是此ind
                                else
                                    allconnBranchInd = find(Data.branch(:,1)== thisNBSLoadPath{m}(connBusInd+1));%在branch中，找到所有起点与该节点下一个节点相同的branch编号
                                    connBranchInd = find(Data.branch(allconnBranchInd,2)==thisNBSLoadPath{m}(connBusInd));%在上述编号中，找到终点与该节点相同的branch编号（唯一）
                                    %connBranchInd = find(Data.branch(:,1)== thisNBSLoadPath{m}(connBusInd+1)&&Data.branch(:,2)==thisNBSLoadPath{m}(connBusInd));%起点是此ind的下一个节点号
                                end
                                ubranch(allconnBranchInd(connBranchInd))=1;
                            end
                            if thisNBSLoadPath{m}(k) == Data.branch(j,2)%路径m的第k个与branchj终点相同
                                connBusInd = k;%在thisNBSLoadPath里，找到与branchj的起点或终点相同的那个节点号所在的位置
                                if thisNBSLoadPath{m}(connBusInd)<thisNBSLoadPath{m}(connBusInd+1)%如果找到的节点号在thisNBSLoadPath里对应的节点号比下一个小
                                    allconnBranchInd = find(Data.branch(:,1)==thisNBSLoadPath{m}(connBusInd));%在branch中，找到所有起点与该节点相同的branch编号
                                    connBranchInd = find(Data.branch(allconnBranchInd,2)==thisNBSLoadPath{m}(connBusInd+1));%在上述branch编号中，找到终点与该节点下一个节点相同的branch编号（唯一）
                                    %connBranchInd = find(Data.branch(:,1)== thisNBSLoadPath{m}(connBusInd)&&Data.branch(:,2)==thisNBSLoadPath{m}(connBusInd+1));%起点是此ind
                                else
                                    allconnBranchInd = find(Data.branch(:,1)==thisNBSLoadPath{m}(connBusInd+1));%在branch中，找到所有起点与该节点下一个节点相同的branch编号
                                    connBranchInd = find(Data.branch(allconnBranchInd,2)==thisNBSLoadPath{m}(connBusInd));%在上述编号中，找到终点与该节点相同的branch编号（唯一）
                                    %connBranchInd = find(Data.branch(:,1)== thisNBSLoadPath{m}(connBusInd+1)&&Data.branch(:,2)==thisNBSLoadPath{m}(connBusInd));%起点是此ind的下一个节点号
                                end
                                ubranch(allconnBranchInd(connBranchInd))=1;
                            end
                        end
                    end
                end
            end
            %                 for j = 1:size(Data.branch,1)
            %                     connBranchInd = [];
            %                     % check which branch is connected with branch j
            %                     for k = 1:size(Data.branch,1)
            %                         if Data.branch(j,1) == Data.branch(k,1) || Data.branch(j,1) == Data.branch(k,2) || Data.branch(j,2) == Data.branch(k,1) || Data.branch(j,2) == Data.branch(k,2)
            %                             connBranchInd = [connBranchInd,k];
            %                         end
            %                     end
            %                     if sum(ubranch(connBranchInd)) >= 1
            %                         ubranch(j) = 1;
            %                     end
            %                 end

            %for each NBS与NBS相连的line恢复，thenNBS恢复(有共同节点)
            for n = 1:numel(priority)
                thisNBSbranchind = Data.branch(:,2)==Data.NBS(priority(n),1);%找到与NBS节点号相同的tbus的位置（该位置是对应的branch的标号）
                if sum(ubranch(thisNBSbranchind)) >= 1
                    ust(priority(n)) = 1;
                else
                    ust(priority(n)) = 0;
                end

                %update the ramping state of the NBS
                P_ramp = Data.NBS(:,18);%从数据中找到P_ramp
                P = Data.NBS(:,2);
                if P(priority(n)) > P_ramp(priority(n))
                    uramp(priority(n)) = 1;
                else
                    uramp(priority(n)) = 0;
                end
            end

            orgCase = case3901;
            updatedCase = orgCase;
            updatedCase.branch(:,11) = ubranch;
            updatedCase.gen(3:end,8) = ust;
            save('newcase.mat',"updatedCase");
            %results = runpf('newcase');
            results = runpf(updatedCase);
            if results.iterations == 10
                results.success
            end
            if results.success == 0
                CV(i) = 10^6;
                continue
            end




            %发电机输出功率约束、爬坡约束
            %从results中提取数据
            P = results.gen(:,2);%有功输出
            Q = results.gen(:,3);%无功输出
            Pmax = results.gen(:,9);%最大输出功率
            %             P_ramp = results.gen(:,18);%爬坡功率
            Qmax = results.gen(:,4);%最大输出无功
            V = results.bus(:,8);%电压幅值
            Vmax = results.bus(:,12);%最大电压
            Vmin = results.bus(:,13);%最小电压
            %BS+NBS输出功率约束
            p1 = zeros(a+b,1);
            for g = 1:a+b
                if Pmax(g)-P(g)>=0 && P(g)>=0 && Qmax(g)-Q(g)>=0 && Q(g)>=0
                    P(g) = P(g);
                else
                    p1(g) = abs(Pmax(g)-P(g))+abs(P(g));
                end
                p1 = sum(p1);
            end
            %NBS爬坡约束
            %             for g = 1:b
            %                 if (P_ramp(g)/10)*T-(P(g,t)-P(g,t-1)) >= 0&&P(g,t)-P(g,t-1) >= 0
            %                     P(g,t) = P(g,t);
            %                 else
            %                     p2 = abs((P_ramp(g)/10)*T-(P(g,t)-P(g,t-1)))+abs(P(g,t)-P(g,t-1));
            %                 end
            %             end

            %潮流约束（针对线路for each branch）
            Smax = Data.branch(:,6);
            PF = results.branch(:,14);
            QF = results.branch(:,15);
            p3 = zeros(46,1);
            for j = 1:size(Data.branch,1)
                if Smax(j)^2*ubranch(j)-(PF(j)^2+QF(j)^2) < 0
                    p3(j) = abs(Smax(j)^2*ubranch(j)-PF(j)^2-QF(j)^2);
                end
                p3 = sum(p3);
            end
            %电压约束
            p4 = zeros(39,1);
            for d = 1:size(Data.bus,1)
                if Vmax(d)-V(d)<0 || V(d)-Vmin(d)<0
                    p4(d) = abs(Vmax(d)-V(d))+abs(V(d)-Vmin(d));
                end
                p4 = sum(p4);
            end

            %等式约束
            %功率平衡约束
            PD = Data.bus(:,3);
            QD = Data.bus(:,4);
            PT = results.branch(:,16);
            QT = results.branch(:,17);
            p2 = zeros(29,1);
            p5 = zeros(29,1);
            for j = 1:29
                if PD(j)>0
                    connbranch1Ind = find(Data.branch(:,1)==j);
                    connbranch2Ind = find(Data.branch(:,2)==j);
                    if PD(j)+sum(PF(connbranch1Ind))+sum(PT(connbranch2Ind)) ~= 0
                        p2(j) = abs(PD(j))+abs(sum(PF(connbranch1Ind)))+abs(sum(PF(connbranch2Ind)));
                    end
                    if QD(j)+sum(QF(connbranch1Ind))+sum(QT(connbranch2Ind)) ~= 0
                        p5(j) = abs(QD(j))+abs(sum(QF(connbranch1Ind)))+abs(sum(QT(connbranch2Ind)));
                    end
                end
                if PD(j) == 0
                    connbranch1Ind = find(Data.branch(:,1)==j);
                    connbranch2Ind = find(Data.branch(:,2)==j);
                    if sum(PF(connbranch1Ind))+sum(PT(connbranch2Ind)) ~= 0
                        p2(j) = abs(sum(PF(connbranch1Ind)))+abs(sum(PF(connbranch2Ind)));
                    end
                    if sum(QF(connbranch1Ind))+sum(QT(connbranch2Ind)) ~= 0
                        p5(j) = abs(sum(QF(connbranch1Ind)))+abs(sum(QT(connbranch2Ind)));
                    end
                end
                p2 = sum(p2);
                p5 = sum(p5);
            end

            %BS启动NBS的功率约束
            Pramp = Data.NBS(:,18);
            PBS = Data.BS(:,17);
            p6 = zeros(size(Data.BS,1),1);
            for j = 1:size(Data.BS,1)
                BSNInd = find(v(j,:)==1);
                BSNInd = BSNInd';
                if sum(Pramp(BSNInd)) > PBS(j)
                    p6(j) = sum(Pramp(BSNInd)) - PBS(j);
                end
                p6 = sum(p6);
            end

            %NBS启动Load的功率约束
            Pout = results.gen([1:2,4:6,8:10],2);
            PL = results.bus([1,3,4,7,8,9,12,15,16,18,20,21,23:29,31,39],3);
            p7 = zeros(size(Data.NBS,1),1);
            for j = 1:size(Data.NBS,1)
                NBSLInd = find(h(j,:)==1);
                NBSLInd = NBSLInd';
                if sum(PL(NBSLInd)) > Pout(j)
                    p7(j) = sum(PL(NBSLInd))-Pout(j);
                end
                p7 = sum(p7);
            end

            % calculate the constraint violation in the following
            CV(i) = p1+p2+p3+p4+p5+p6+p7;
        end

        % collect accessory data for understanding the solutions in the
        % following (optional)

        output1 = Objs; % matrix for saving objective function values
        output2 = CV; % matrix for saving constraint violation values (optional)
        output3 = []; % matrix or cells for saving accessory data (optional), a solution's accessory data should be saved in a row
        
        %output1 = output1 + output2
end

if ~exist('output2','var')
    output2 = [];
elseif ~exist('output3','var')
    output3 = [];
end
end