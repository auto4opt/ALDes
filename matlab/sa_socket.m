try
    % 创建一个 TCP/IP 服务器对象，监听所有 IP 地址上的端口 30000
    t = tcpip('0.0.0.0', 30000, 'NetworkRole', 'server');
    t.Timeout = 10;  % 设置超时时间为 10 秒

    % 打开 TCP/IP 连接，等待客户端连接
    fopen(t);
    disp('MATLAB server is waiting for connection...');

    % 无限循环以处理客户端请求
    while true
        % 检查是否有可用数据
        if t.BytesAvailable > 0
            % 从连接中读取所有可用数据
            data = fread(t, t.BytesAvailable);
            
            % 将读取的数据转换为字符串
            command = char(data');
            % 解析接收到的整数
            numbers = str2double(strsplit(command, ','));
            T = numbers(1);
            problem_id = numbers(2);
            problem_dim = numbers(3);
            % 打印接收到的命令（仅用于调试）
            disp(['Received command: ', command]);
            
            % 如果接收到 'exit' 命令，则退出循环
            if strcmp(command, 'exit')
                break;
            else
                % 计算命令结果（注意：此处使用 eval 是不安全的，仅用于示例）
                %result = eval(command);
                AutoOpt('Mode','solve','Problem','pbo','InstanceSolve',[problem_dim],'AlgName',"Discrete Simulated Annealing",'inital_t',T,'ProbN',50,'ProbFE',50000,'AlgRuns',1,'problem_id',problem_id);
                load(strcat("Discrete Simulated Annealing",int2str(problem_id)));
                res =[];
                [x, y] = size(bestSolutions);
                for ii =1: y
                    res(ii)=cat(1,bestSolutions(1,ii).fit);
                end
                % 将结果转换为字符串并发送回客户端
                fwrite(t, num2str(median(res)));
                disp(median(res));
            end
        else
            % 暂停片刻以避免忙等待
            pause(0.1);
        end
    end
catch ME
    disp('An error occurred:');
    disp(ME.message);
end

% 确保在任何情况下都能关闭连接并清理对象
if exist('t', 'var') && isvalid(t)
    fclose(t);
    delete(t);
    clear t;
end
disp('MATLAB server has stopped.');
