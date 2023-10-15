from Configurations import Environment as envProps
from Environment.Player import Player
from Environment.Ball import Ball
from Utilities.Logger import Logger
from gymnasium import spaces
import numpy as np
import os

class Environment_1A_0D_0K():
		def __init__(self):
				self.environment = "1A_0D_0K.xml"
				self.set_env_path(self.environment)

				self.home_team_size = 1
				self.away_team_size = 0

				# Action Space
				self.action_space = spaces.Box(
						low=np.array(
								[envProps.BOLT_MIN_ROTATION, envProps.BOLT_MIN_SPEED], dtype=np.float32
						),
						high=np.array(
								[envProps.BOLT_MAX_ROTATION, envProps.BOLT_MAX_SPEED], dtype=np.float32
						),
				)

				# Observation Space
				player_state_space_low = np.array(
						[
								-envProps.FIELD_LENGTH,
								-envProps.FIELD_WIDTH,
								envProps.BOLT_MIN_SPEED,
								envProps.BOLT_MIN_ROTATION,
						],
						dtype=np.float32,
				)
				player_state_space_high = np.array(
						[
								envProps.FIELD_LENGTH,
								envProps.FIELD_WIDTH,
								envProps.BOLT_MAX_SPEED,
								envProps.BOLT_MAX_ROTATION,
						],
						dtype=np.float32,
				)
				ball_state_space_low = np.array(
						[
								-envProps.FIELD_LENGTH,
								-envProps.FIELD_WIDTH,
								envProps.BALL_MIN_SPEED,
								envProps.BALL_MIN_ROTATION,
						],
						dtype=np.float32,
				)
				ball_state_space_high = np.array(
						[
								envProps.FIELD_LENGTH,
								envProps.FIELD_WIDTH,
								envProps.BALL_MAX_SPEED,
								envProps.BALL_MAX_ROTATION,
						],
						dtype=np.float32,
				)

				self.observation_space = spaces.Box(
						low=np.concatenate((player_state_space_low, ball_state_space_low)),
						high=np.concatenate((player_state_space_high, ball_state_space_high)),
						dtype=np.float32,
				)

				self.logger = Logger()

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

		def get_observation_space(self, data):
				player_position = self.player.get_position(data)
				player_velocity = self.player.get_velocity(data)[:2]
				player_speed, player_rotation = self.get_speed_and_rotation_from_velocities(
						player_velocity[0], player_velocity[1]
				)

				ball_position = self.ball.get_position(data)
				ball_velocity = self.ball.get_velocity(data)[:2]
				ball_speed, ball_rotation = self.get_speed_and_rotation_from_velocities(
						ball_velocity[0], ball_velocity[1]
				)

				player_state = np.concatenate(
						(player_position[:2], player_speed, player_rotation)
				)
				ball_state = np.concatenate((ball_position[:2], ball_speed, ball_rotation))

				state_space = np.concatenate((player_state, ball_state))
				return state_space

		def get_speed_and_rotation_from_velocities(self, x_vel, y_vel):
				speed = np.sqrt(x_vel**2 + y_vel**2)
				rotation = ((np.arctan2(y_vel, x_vel)) + 360) % 360
				return [speed], [rotation]

		def preprocess_sigmoid_actions(self, action):
				# Actions are in the range [0, 1]
				action_rotation, action_speed = action
				player_rotation = action_rotation * 359
				player_speed = action_speed * 20
				return player_rotation, player_speed

		def preprocess_tanh_actions(self, action):
				# Actions are in the range [-1, 1]
				# Linear scaling: f(x) = 0.5 * (max - min) * x + 0.5 * (max + min)
				action_rotation, action_speed = action
				player_rotation = (0.5 * 359 * action_rotation) + (0.5 * 359)
				player_speed = (0.5 * 20 * action_speed) + (0.5 * 20)
				return player_rotation, player_speed

		def step(self, data, action):
				# Apply action on player and get reward
				rotation, speed = self.preprocess_tanh_actions(action)
				self.player.set_heading_and_velocity(data, rotation, speed)
				reward = self.player.get_reward(data, self.ball)

				# Get new observation
				new_observation = self.get_observation_space(data)

				# Check if ball is kicked
				done = False
				contacts = data.contact
				for contact in contacts:
					if contact.geom1 == self.player.id_geom and contact.geom2 == self.ball.id_geom:
						done = True	
				
				info = {}

				return new_observation, reward, done, info
