load('20230724T151736_episode_record.mat')

performance = [episode_record.performance];

reward = [episode_record.reward];

X = size(reward,2);

X = 1:X;

plot(X,performance,X,reward)

plot(performance)