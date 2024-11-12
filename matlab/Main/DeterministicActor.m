actorNet =[ 
    featureInputLayer(15,Name="comPathIn")
    selfAttentionLayer(5,15)
    layerNormalizationLayer
    fullyConnectedLayer(15)
    reluLayer;
    scalingLayer(Name="meanPathOut",Scale=ActionInfo.UpperLimit)];


actorNet = dlnetwork(actorNet);
summary(actorNet)
%rlContinuousGaussianActor

actor = rlContinuousDeterministicActor(actorNet1, ObservationInfo, ActionInfo);
actor.UseDevice = "gpu";

% test
result = getAction(actor,{rand(ObservationInfo.Dimension)});
x = rand(15,1);
dlx = dlarray(x,'CB');
result = forward(actorNet1,dlx);%}