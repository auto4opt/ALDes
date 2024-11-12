function [env] = env()
%ENV creat env
%   此处显示详细说明

ObservationInfo = rlNumericSpec([15 1]);
ObservationInfo.Name = "States";
ObservationInfo.Description = 'done';

% continue 15 dim action
ActionInfo = rlNumericSpec([15 1]);
ActionInfo.Name = "Action";
ActionInfo.LowerLimit=0;
ActionInfo.UpperLimit=1;

env = rlFunctionEnv(ObservationInfo,ActionInfo,"myStepFunction","myResetFunction");
end

