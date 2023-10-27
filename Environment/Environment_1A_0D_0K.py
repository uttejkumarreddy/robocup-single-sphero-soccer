from Configurations import Environment as envProps
from Environment.Player import Player
from Environment.Ball import Ball
from Utilities.Logger import Logger
from gymnasium import spaces
import numpy as np
import os

class Environment_1A_0D_0K():
		def __init__(self):
				self.environment = "1A_0D_0K/extrasmall.xml"
				self.set_env_path(self.environment)

				self.home_team_size = 1
				self.away_team_size = 0

				# Action Space - velocity (0-20), rotation (0-359)
				self.action_space = spaces.Box(
						low  = np.array([envProps.ENV_BOLT_MIN_SPEED, envProps.ENV_BOLT_MIN_ROTATION], dtype=np.int32),
						high = np.array([envProps.ENV_BOLT_MAX_SPEED, envProps.ENV_BOLT_MAX_ROTATION], dtype=np.int32),
				)

				# Observation Space
				observation_space_low = np.array([
					-envProps.FIELD_LENGTH, # agent x position
					-envProps.FIELD_WIDTH, # agent y position
					envProps.ENV_BOLT_MIN_SPEED, # agent velocity
					envProps.ENV_BOLT_MIN_ROTATION, # agent rotation
					-envProps.FIELD_LENGTH, # ball x position
					-envProps.FIELD_WIDTH, # ball y position
					envProps.ENV_BALL_MIN_SPEED, # ball velocity
					envProps.ENV_BALL_MIN_ROTATION, # ball rotation
				], dtype=np.float32,)

				observation_space_high = np.array([
					envProps.FIELD_LENGTH, # agent x position
					envProps.FIELD_WIDTH, # agent y position
					envProps.ENV_BOLT_MAX_SPEED, # agent velocity
					envProps.ENV_BOLT_MAX_ROTATION, # agent rotation
					envProps.FIELD_LENGTH, # ball x position
					envProps.FIELD_WIDTH, # ball y position
					envProps.ENV_BALL_MAX_SPEED, # ball velocity
					envProps.ENV_BALL_MAX_ROTATION, # ball rotation
				], dtype=np.float32,)

				self.observation_space = spaces.Box(
						low = observation_space_low,
						high= observation_space_high,
						dtype=np.float32,
				)

				self.logger = Logger()
				self.logger.write('Observation space: {0}'.format(self.observation_space))
				self.logger.write('Action space: {0}'.format(self.action_space))

		def set_env_path(self, environment):
				dirname = os.path.dirname(__file__)
				abspath = os.path.join(dirname + "/../Mujoco/" + environment)
				self.env_path = abspath

		def get_env_path(self):
				return self.env_path

		def init_characters(self, model):
				# Players - Single Player
				self.player = Player(
						model,
						envProps.TEAM_HOME,
						"home_player_1",
						self.observation_space,
						self.action_space,
				)

				# Ball
				self.ball = Ball(model, envProps.NAME_BALL)

		def get_home_team_size(self):
				return self.home_team_size

		def get_away_team_size(self):
				return self.away_team_size
			
		def get_speed_from_velocity(self, velocity):
				x_vel, y_vel = velocity[0], velocity[1]
				speed = np.sqrt(x_vel**2 + y_vel**2)
				return [speed]

		def get_heading_from_velocity(self, velocity):
				x_vel, y_vel = velocity[0], velocity[1]
				rotation = ((np.arctan2(y_vel, x_vel)) + 360) % 360
				return [rotation]

		def get_observation_space(self, data):
				player_position = self.player.get_position(data)
				player_velocity = self.player.get_velocity(data)[:2]
				player_speed, player_rotation = self.get_speed_from_velocity(player_velocity), self.get_heading_from_velocity(player_velocity)

				ball_position = self.ball.get_position(data)
				ball_velocity = self.ball.get_velocity(data)[:2]
				ball_speed, ball_rotation = self.get_speed_from_velocity(ball_velocity), self.get_heading_from_velocity(ball_velocity)

				player_state = np.concatenate((player_position[:2], player_speed, player_rotation))
				ball_state = np.concatenate((ball_position[:2], ball_speed, ball_rotation))

				state_space = np.concatenate((player_state, ball_state))
				return state_space

		def preprocess_sigmoid_actions(self, action):
				# Actions are in the range [0, 1]
				action_speed, action_rotation = action
				player_rotation = action_rotation * 359
				player_speed = action_speed * 20
				return player_speed, player_rotation

		def scale_linear(self, x, min, max):
			# Linear scaling: f(x) = 0.5 * (max - min) * x + 0.5 * (max + min)
			return (0.5 * (max - min) * x) + (0.5 * (max + min))

		def preprocess_tanh_actions(self, action):
				# Actions are in the range [-1, 1]
				action_speed, action_rotation = action
				player_speed = self.scale_linear(action_speed, envProps.ENV_BOLT_MIN_SPEED, envProps.ENV_BOLT_MAX_SPEED) # Scale speed to [0-20]
				player_rotation = self.scale_linear(action_rotation, envProps.ENV_BOLT_MIN_ROTATION, envProps.ENV_BOLT_MAX_ROTATION) # Scale rotation to [0-359]
				return player_speed, player_rotation

		def step(self, data, action):
				# Apply action on player and get reward
				speed, rotation = self.preprocess_tanh_actions(action) # ([0-20], [0-359])
				self.player.set_heading_and_velocity(data, rotation, speed)
				reward = self.player.get_reward(data, self.ball)

				# Get new observation
				new_observation = self.get_observation_space(data)

				# Check if ball is kicked
				done = False
				contacts = data.contact
				for c in contacts:
					if c.geom1 == self.ball.id_geom and c.geom2 == self.player.id_geom:
						done = True	
				
				info = {}

				return new_observation, reward, done, info
