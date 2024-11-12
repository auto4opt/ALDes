global Problem Data Setting seedTrain;

global episode_record best_action best_p;

global episode_ite;

global times;

times = datestr(now,30);

episode_ite =0;
episode_record = struct;
best_p = 0;
rng(0);

[Problem,Data,Setting,seedTrain] = opt_env('Mode','design','Problem','CEC2005_f1','InstanceTrain',[1,2],'InstanceTest',3);

% get env
env = env();

% get net
ObservationInfo = getObservationInfo(env);
ActionInfo = getActionInfo(env);

% bug here? 等号右侧的输出数目不足，不满足赋值要求。
[critic,actor] = net(ObservationInfo,ActionInfo);


% set agent parameters
criticOptions = rlOptimizerOptions( ...
    LearnRate=1e-3, ...
    GradientThreshold=1);
actorOptions = rlOptimizerOptions( ...
    LearnRate=2e-4, ...
    GradientThreshold=1);

agentOpts = rlPPOAgentOptions(...
    SampleTime=-1,...
    ActorOptimizerOptions=actorOptions,...
    CriticOptimizerOptions=criticOptions,...
    ExperienceHorizon=200,...
    ClipFactor=0.2,... 
    EntropyLossWeight=0.01,...
    MiniBatchSize=64,...
    NumEpoch=3,...
    AdvantageEstimateMethod="gae",...
    GAEFactor=0.95,...
    DiscountFactor=0.998);


agent = rlPPOAgent(actor,critic,agentOpts);


% Test
%getAction(agent,{rand(ObservationInfo.Dimension)});

% train parameters
trainOpts = rlTrainingOptions(...
    MaxEpisodes=100000,...
    MaxStepsPerEpisode=1,...
    ScoreAveragingWindowLength=10,...
    Plots="training-progress",...
    StopTrainingCriteria="AverageReward",...
    StopTrainingValue=10000);

doTraining = true;


if doTraining
    trainingStats = train(agent,env,trainOpts);
    save([times ,'_Agent.mat'],'agent');
    save([times ,'_best_action.mat'],'best_action');
    save([times ,'_episode_record.mat'],'episode_record');
else
    load('Agent.mat','agent');
end