function [critic,actor] = net(ObservationInfo,ActionInfo)
%NET 此处显示有关此函数的摘要
%   此处显示详细说明

% actor
% Define common input path layer
commonPath = [ 
    featureInputLayer(15,Name="comPathIn")
    selfAttentionLayer(5,15)
    layerNormalizationLayer
    fullyConnectedLayer(15,'WeightsInitializer','orthogonal')
    tanhLayer(Name="comPathOut");];

% Define mean value path
meanPath = [
    fullyConnectedLayer(15,'WeightsInitializer','orthogonal',Name="meanPathIn")
    tanhLayer
    fullyConnectedLayer(prod(ActionInfo.Dimension),'WeightsInitializer','orthogonal',Name="meanPathOut");
    ];

% Define standard deviation path
sdevPath = [
    fullyConnectedLayer(15,'WeightsInitializer','orthogonal',"Name","stdPathIn")
    tanhLayer
    fullyConnectedLayer(prod(ActionInfo.Dimension),'WeightsInitializer','orthogonal');
    softplusLayer(Name="stdPathOut") 
    ];

% Add layers to layerGraph object
actorNet = layerGraph(commonPath);
actorNet = addLayers(actorNet,meanPath);
actorNet = addLayers(actorNet,sdevPath);

% Connect paths
actorNet = connectLayers(actorNet,"comPathOut","meanPathIn/in");
actorNet = connectLayers(actorNet,"comPathOut","stdPathIn/in");


actorNet = dlnetwork(actorNet);
summary(actorNet)
%plot(actorNet)

actor = rlContinuousGaussianActor(actorNet, ObservationInfo, ActionInfo, ...
    "ActionMeanOutputNames","meanPathOut",...
    "ActionStandardDeviationOutputNames","stdPathOut",...
    ObservationInputNames="comPathIn");

actor.UseDevice = "gpu";

%test
%x = rand(15,1);
%dlx = dlarray(x,'CB');
%[comPathOut meanPathOut stdPathOut] = forward(actorNet,dlx,Outputs=["comPathOut" "meanPathOut" "stdPathOut"])


% critic  
criticNet = [
    featureInputLayer(15)
    fullyConnectedLayer(50)
    tanhLayer
    fullyConnectedLayer(25)
    tanhLayer
    fullyConnectedLayer(1)];

criticNet = dlnetwork(criticNet);
summary(criticNet)
%plot(criticNet)

critic = rlValueFunction(criticNet,ObservationInfo);
critic.UseDevice = "gpu";


%test
%getValue(critic,{rand(ObservationInfo.Dimension)})

end

