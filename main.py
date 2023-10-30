import Environment

MAX_EPISODES = 100000

env = Environment.make('1A_0D_0K')
for episode_count in range(1, MAX_EPISODES + 1):
	env.run(episode_count)