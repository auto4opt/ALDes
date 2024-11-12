function [InitialObservation, LoggedSignal] = myResetFunction()
% Reset function to place custom  environment into a random
% initial state.

LoggedSignal.State = [1;1;1;1;1;1;1;1;1;1;1;1;1;1;1];
%LoggedSignal.State = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
InitialObservation = LoggedSignal.State;

end
