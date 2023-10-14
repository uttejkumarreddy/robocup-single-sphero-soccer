import Environment

from AI.DDPG.Agent import Agent
from AI.DDPG.ReplayBuffer import ReplayBuffer

MAX_EPISODES = 100000

env = Environment.make('1A_0D_0K')
for episode_count in range(MAX_EPISODES):
	env.run(episode_count)

	# Save models every 20 episodes
	if episode_count % 20 == 0:
		env.save_models()