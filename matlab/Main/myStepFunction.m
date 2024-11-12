function [NextObs,Reward,IsDone,LoggedSignals] = myStepFunction(Action,LoggedSignals)

global Problem Data Setting seedTrain;
global episode_record best_action best_p;
global episode_ite;
global times;

episode_ite = episode_ite +1;

Action = sigmoid(Action);
IsDone = true;

obj = DESIGN;
Setting.Action = Action;
[performance Algs change] = obj.Get_performance(Problem,Data,Setting,seedTrain);
%performance = Get_p('Mode','design','Problem','CEC2005_f1','InstanceTrain',[1,2],'InstanceTest',3);

baseline =0;

if episode_ite>10
    per = [episode_record.performance];
    per = per(end-9:end);
    rew =0;
    for i=1:10
        rew = rew + per_to_rew(per(i));
    end
    
    baseline = rew/10;
end


Reward = per_to_rew(performance);
Reward =Reward -baseline - change*10;


episode_record(episode_ite).performance = performance;
episode_record(episode_ite).reward =Reward;
episode_record(episode_ite).Algs = Algs;

per = [episode_record.performance];
min_p = min(per);

if performance<=min_p
    episode_ite
    performance
    best_action = Action;
    Reward = Reward*1.5;
    save([times ,'_episode_record.mat'],'episode_record');
end


LoggedSignals.State = [1;1;1;1;1;1;1;1;1;1;1;1;1;1;1];
NextObs = LoggedSignals.State;

end

function [reward] = per_to_rew(performance)
    distance = performance - (-450);
    reward = ((1/distance)*450 - 1)*10;
end