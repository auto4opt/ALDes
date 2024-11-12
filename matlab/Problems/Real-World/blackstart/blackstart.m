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
            BSNBSInd = zeros(1,b);  %�ĸ�BS�����ĸ�NBS����Ӧ��λ�ã�
            BSNBSPath = Data.BSNBSPath;%BS to NBS��·��
            thisBSNBSPath = cell(b,1);%���ݽ⣬��ȡ��BStoNBS��·��
            NBSLoadInd = zeros(1,c);%�ĸ�NBS�����ĸ�Load����Ӧ��λ�ã�
            NBSLoadPath = Data.NBSLoadPath;%NBS to Load��·��
            thisNBSLoadPath = cell(c,1);%���ݽ⣬��ȡ��NBStoLoad��·��
            % ����solution����ȡBS��NBS��·��
            for j = 1:size(v,2)
                BSNBSInd(j) = find(v(:,j)==1);
                thisBSNBSPath{j} =  BSNBSPath{BSNBSInd(j),j};
            end
            % ����solution����ȡNBS��Load��·��
            for j = 1:size(h,2)
                NBSLoadInd(j) = find(h(:,j)==1);
                thisNBSLoadPath{j} =  NBSLoadPath{NBSLoadInd(j),j};
            end

            %���¸�����״̬
            ubranch = zeros(size(Data.branch,1),1);% branch��״̬
            ust = zeros(b,1); % starting state of NBS
            uramp = zeros(b,1); % rumping state of NBS
            %             ubus = zeros(size(Data.bus,1),1);
            %             ubus(30:31) = 1;
            %BS�ָ���then��BSֱ��������line�ָ�(��BS�ڵ��һ����line)
            for j = 1:size(Data.BS,1)
                thisBSbranchind = Data.branch(:,2)==Data.BS(j,1);%�ҵ���BS�ڵ����ȵ�tbus��λ�ã����λ��Ҳ����branch�ı�ţ�
                ubranch(thisBSbranchind) = 1;
            end

            %��line�����߻ָ������ָܻ�(�������й�ͬ�Ľڵ�)
            %����NBS���ȼ�������BStoNBS��each·��������״̬
            for m = 1:numel(priority)
                for k = 1:numel(thisBSNBSPath{priority(m)}(1:end-1))
                    for j = 1:size(Data.branch,1)
                        if ubranch(j) == 1
                            if thisBSNBSPath{priority(m)}(k) == Data.branch(j,1)
                                connBusInd = k;%��thisBSNBSPath���ҵ���branchj�й�ͬ�ڵ�Ľڵ�ŵ�λ�ã���thisBSNBSPath��priority��m���еĵڼ�����
                                if thisBSNBSPath{priority(m)}(connBusInd)<thisBSNBSPath{priority(m)}(connBusInd+1)%���ݽڵ�ţ��ҵ�thisBSNBSPath�дνڵ������һ�ڵ����ɵ�branch��ind
                                    allconnBranchInd = find(Data.branch(:,1)==thisBSNBSPath{priority(m)}(connBusInd));%���ݽڵ�ţ���branch��������ҵ���ýڵ����ͬ������branch�ı��
                                    connBranchInd = find(Data.branch(allconnBranchInd,2)==thisBSNBSPath{priority(m)}(connBusInd+1));%���ҵ��ı�����branch���յ����ҳ�����һ���ڵ����ͬ��branch�ı��
                                    %connBranchInd = find(Data.branch(:,1)==thisBSNBSPath{priority(m)}(connBusInd)&&Data.branch(:,2)==thisBSNBSPath{priority(m)}(connBusInd+1));%�ҵ��Ľڵ�ű�����������Ľڵ��С
                                else
                                    allconnBranchInd = find(Data.branch(:,1)==thisBSNBSPath{priority(m)}(connBusInd+1));
                                    connBranchInd = find(Data.branch(allconnBranchInd,2)==thisBSNBSPath{priority(m)}(connBusInd));
                                    %connBranchInd = find(Data.branch(:,1)==thisBSNBSPath{priority(m)}(connBusInd+1)&&Data.branch(:,2)==thisBSNBSPath{priority(m)}(connBusInd));%�ҵ��Ľڵ�ű�����������Ľڵ�Ŵ�
                                end
                                ubranch(allconnBranchInd(connBranchInd))=1;
                                restoredbranchInd = {restoredbranchInd,allconnBranchInd(connBranchInd)};
                            end
                            if thisBSNBSPath{priority(m)}(k) == Data.branch(j,2)
                                connBusInd = k;%��thisBSNBSPath���ҵ���branchj�й�ͬ�ڵ�Ľڵ�ŵ�λ�ã���thisBSNBSPath��priority��m���еĵڼ�����
                                if thisBSNBSPath{priority(m)}(connBusInd)<thisBSNBSPath{priority(m)}(connBusInd+1)%���ݽڵ�ţ��ҵ�thisBSNBSPath�дνڵ������һ�ڵ����ɵ�branch��ind
                                    allconnBranchInd = find(Data.branch(:,1)==thisBSNBSPath{priority(m)}(connBusInd));
                                    connBranchInd = find(Data.branch(allconnBranchInd,2)==thisBSNBSPath{priority(m)}(connBusInd+1));
                                    %connBranchInd = find(Data.branch(:,1)==thisBSNBSPath{priority(m)}(connBusInd)&&Data.branch(:,2)==thisBSNBSPath{priority(m)}(connBusInd+1));%�ҵ��Ľڵ�ű�����������Ľڵ��С
                                else
                                    allconnBranchInd = find(Data.branch(:,1)==thisBSNBSPath{priority(m)}(connBusInd+1));
                                    connBranchInd = find(Data.branch(allconnBranchInd,2)==thisBSNBSPath{priority(m)}(connBusInd));
                                    %connBranchInd = find(Data.branch(:,1)==thisBSNBSPath{priority(m)}(connBusInd+1)&&Data.branch(:,2)==thisBSNBSPath{priority(m)}(connBusInd));%�ҵ��Ľڵ�ű�����������Ľڵ�Ŵ�
                                end
                                ubranch(allconnBranchInd(connBranchInd))=1;
                                restoredbranchInd = {restoredbranchInd,allconnBranchInd(connBranchInd)};
                            end
                        end
                    end
                end
            end

            %����NBStoLoad��״̬
            for m = 1:size(thisNBSLoadPath,1)
                for k = 1:numel(thisNBSLoadPath{m}(1:end-1))
                    for j = 1:size(Data.branch,1)
                        if ubranch(j) == 1
                            if thisNBSLoadPath{m}(k) == Data.branch(j,1)%·��m�ĵ�k����branchj�����ͬ
                                connBusInd = k;%��thisNBSLoadPath��ҵ���branchj�������յ���ͬ���Ǹ��ڵ�����ڵ�λ��
                                if thisNBSLoadPath{m}(connBusInd)<thisNBSLoadPath{m}(connBusInd+1)%����ҵ��Ľڵ����thisNBSLoadPath���Ӧ�Ľڵ�ű���һ��С
                                    allconnBranchInd = find(Data.branch(:,1)== thisNBSLoadPath{m}(connBusInd));%��branch�У��ҵ����������ýڵ���ͬ��branch���
                                    connBranchInd = find(Data.branch(allconnBranchInd,2)==thisNBSLoadPath{m}(connBusInd+1));%������branch����У��ҵ��յ���ýڵ���һ���ڵ���ͬ��branch��ţ�Ψһ��
                                    %connBranchInd = find(Data.branch(:,1)== thisNBSLoadPath{m}(connBusInd)&&Data.branch(:,2)==thisNBSLoadPath{m}(connBusInd+1));%����Ǵ�ind
                                else
                                    allconnBranchInd = find(Data.branch(:,1)== thisNBSLoadPath{m}(connBusInd+1));%��branch�У��ҵ����������ýڵ���һ���ڵ���ͬ��branch���
                                    connBranchInd = find(Data.branch(allconnBranchInd,2)==thisNBSLoadPath{m}(connBusInd));%����������У��ҵ��յ���ýڵ���ͬ��branch��ţ�Ψһ��
                                    %connBranchInd = find(Data.branch(:,1)== thisNBSLoadPath{m}(connBusInd+1)&&Data.branch(:,2)==thisNBSLoadPath{m}(connBusInd));%����Ǵ�ind����һ���ڵ��
                                end
                                ubranch(allconnBranchInd(connBranchInd))=1;
                            end
                            if thisNBSLoadPath{m}(k) == Data.branch(j,2)%·��m�ĵ�k����branchj�յ���ͬ
                                connBusInd = k;%��thisNBSLoadPath��ҵ���branchj�������յ���ͬ���Ǹ��ڵ�����ڵ�λ��
                                if thisNBSLoadPath{m}(connBusInd)<thisNBSLoadPath{m}(connBusInd+1)%����ҵ��Ľڵ����thisNBSLoadPath���Ӧ�Ľڵ�ű���һ��С
                                    allconnBranchInd = find(Data.branch(:,1)==thisNBSLoadPath{m}(connBusInd));%��branch�У��ҵ����������ýڵ���ͬ��branch���
                                    connBranchInd = find(Data.branch(allconnBranchInd,2)==thisNBSLoadPath{m}(connBusInd+1));%������branch����У��ҵ��յ���ýڵ���һ���ڵ���ͬ��branch��ţ�Ψһ��
                                    %connBranchInd = find(Data.branch(:,1)== thisNBSLoadPath{m}(connBusInd)&&Data.branch(:,2)==thisNBSLoadPath{m}(connBusInd+1));%����Ǵ�ind
                                else
                                    allconnBranchInd = find(Data.branch(:,1)==thisNBSLoadPath{m}(connBusInd+1));%��branch�У��ҵ����������ýڵ���һ���ڵ���ͬ��branch���
                                    connBranchInd = find(Data.branch(allconnBranchInd,2)==thisNBSLoadPath{m}(connBusInd));%����������У��ҵ��յ���ýڵ���ͬ��branch��ţ�Ψһ��
                                    %connBranchInd = find(Data.branch(:,1)== thisNBSLoadPath{m}(connBusInd+1)&&Data.branch(:,2)==thisNBSLoadPath{m}(connBusInd));%����Ǵ�ind����һ���ڵ��
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

            %for each NBS��NBS������line�ָ���thenNBS�ָ�(�й�ͬ�ڵ�)
            for n = 1:numel(priority)
                thisNBSbranchind = Data.branch(:,2)==Data.NBS(priority(n),1);%�ҵ���NBS�ڵ����ͬ��tbus��λ�ã���λ���Ƕ�Ӧ��branch�ı�ţ�
                if sum(ubranch(thisNBSbranchind)) >= 1
                    ust(priority(n)) = 1;
                else
                    ust(priority(n)) = 0;
                end

                %update the ramping state of the NBS
                P_ramp = Data.NBS(:,18);%���������ҵ�P_ramp
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




            %������������Լ��������Լ��
            %��results����ȡ����
            P = results.gen(:,2);%�й����
            Q = results.gen(:,3);%�޹����
            Pmax = results.gen(:,9);%����������
            %             P_ramp = results.gen(:,18);%���¹���
            Qmax = results.gen(:,4);%�������޹�
            V = results.bus(:,8);%��ѹ��ֵ
            Vmax = results.bus(:,12);%����ѹ
            Vmin = results.bus(:,13);%��С��ѹ
            %BS+NBS�������Լ��
            p1 = zeros(a+b,1);
            for g = 1:a+b
                if Pmax(g)-P(g)>=0 && P(g)>=0 && Qmax(g)-Q(g)>=0 && Q(g)>=0
                    P(g) = P(g);
                else
                    p1(g) = abs(Pmax(g)-P(g))+abs(P(g));
                end
                p1 = sum(p1);
            end
            %NBS����Լ��
            %             for g = 1:b
            %                 if (P_ramp(g)/10)*T-(P(g,t)-P(g,t-1)) >= 0&&P(g,t)-P(g,t-1) >= 0
            %                     P(g,t) = P(g,t);
            %                 else
            %                     p2 = abs((P_ramp(g)/10)*T-(P(g,t)-P(g,t-1)))+abs(P(g,t)-P(g,t-1));
            %                 end
            %             end

            %����Լ���������·for each branch��
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
            %��ѹԼ��
            p4 = zeros(39,1);
            for d = 1:size(Data.bus,1)
                if Vmax(d)-V(d)<0 || V(d)-Vmin(d)<0
                    p4(d) = abs(Vmax(d)-V(d))+abs(V(d)-Vmin(d));
                end
                p4 = sum(p4);
            end

            %��ʽԼ��
            %����ƽ��Լ��
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

            %BS����NBS�Ĺ���Լ��
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

            %NBS����Load�Ĺ���Լ��
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