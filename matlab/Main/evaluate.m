load('Agent.mat','agent');
input = [1;1;1;1;1;1;1;1;1;1;1;1;1;1;1];
res = zeros(1,1000);
for i =1:1000
    action = getAction(agent,input);
    action = action{1};
    action = sigmoid(action);
    act1 = action(1:7,:);
    act2 = action(9:15,:);
    res(i) = sum(act1)-sum(act2);
end
plot(res)
mean(res)
