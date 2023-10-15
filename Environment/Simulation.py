from Configurations import Environment as envProps
from Utilities.Logger import Logger

from mujoco.glfw import glfw
import mujoco as mj

import os
import numpy as np


class Simulation:
		def __init__(self, env, sim_length=45, frames_per_second=20):
				self.env = env
				self.logger = Logger()

				self.sim_length = sim_length
				self.frames_per_second = frames_per_second

				self.button_left = False
				self.button_middle = False
				self.button_right = False
				self.lastx = 0
				self.lasty = 0

				self.model = mj.MjModel.from_xml_path(self.env.get_env_path())
				self.data = mj.MjData(self.model)
				self.cam = mj.MjvCamera()
				self.opt = mj.MjvOption()

				self.env.init_characters(self.model)

				glfw.init()
				self.window = glfw.create_window(1200, 900, "RoboCup Soccer - SSL", None, None)
				glfw.make_context_current(self.window)
				glfw.swap_interval(1)

				mj.mjv_defaultCamera(self.cam)
				mj.mjv_defaultOption(self.opt)

				glfw.set_key_callback(self.window, self.keyboard)
				glfw.set_cursor_pos_callback(self.window, self.mouse_move)
				glfw.set_mouse_button_callback(self.window, self.mouse_button)
				glfw.set_scroll_callback(self.window, self.mouse_scroll)

				self.scene = mj.MjvScene(self.model, maxgeom=10000)
				self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

		def keyboard(self, window, key, scancode, act, mods):
				if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
						mj.mj_resetData(self.model, self.data)
						mj.mj_forward(self.model, self.data)

		def mouse_button(self, window, button, act, mods):
				self.button_left = (
						glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
				)
				self.button_middle = (
						glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
				)
				self.button_right = (
						glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
				)

				glfw.get_cursor_pos(window)

		def mouse_move(self, window, xpos, ypos):
				dx = xpos - self.lastx
				dy = ypos - self.lasty
				self.lastx = xpos
				self.lasty = ypos

				if (
						(not self.button_left)
						and (not self.button_middle)
						and (not self.button_right)
				):
						return

				width, height = glfw.get_window_size(window)

				PRESS_LEFT_SHIFT = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
				PRESS_RIGHT_SHIFT = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
				mod_shift = PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT

				if self.button_right:
						if mod_shift:
								action = mj.mjtMouse.mjMOUSE_MOVE_H
						else:
								action = mj.mjtMouse.mjMOUSE_MOVE_V
				elif self.button_left:
						if mod_shift:
								action = mj.mjtMouse.mjMOUSE_ROTATE_H
						else:
								action = mj.mjtMouse.mjMOUSE_ROTATE_V
				else:
						action = mj.mjtMouse.mjMOUSE_ZOOM

				mj.mjv_moveCamera(
						self.model, action, dx / height, dy / height, self.scene, self.cam
				)

		def mouse_scroll(self, window, xoffset, yoffset):
				action = mj.mjtMouse.mjMOUSE_ZOOM
				mj.mjv_moveCamera(
						self.model, action, 0.0, -0.05 * yoffset, self.scene, self.cam
				)

		def reset(self):
				mj.mj_resetData(self.model, self.data)
				mj.mj_kinematics(self.model, self.data)
				mj.mj_forward(self.model, self.data)

				self.init_controller(self.model, self.data)

		def init_controller(self, model, data):
				if envProps.RANDOMIZE_INITIAL_POSITIONS_PLAYERS == True:
						random_position = (
								np.random.uniform(-envProps.FIELD_LENGTH, envProps.FIELD_LENGTH),
								np.random.uniform(-envProps.FIELD_WIDTH, envProps.FIELD_WIDTH),
								envProps.RADIUS_PLAYER,
						)
						self.env.player.set_position(data, random_position)

				if envProps.RANDOMIZE_INITIAL_POSITIONS_BALL == True:
						random_position = (
								np.random.uniform(-envProps.FIELD_LENGTH, envProps.FIELD_LENGTH),
								np.random.uniform(-envProps.FIELD_WIDTH, envProps.FIELD_WIDTH),
								envProps.RADIUS_BALL,
						)
						self.env.ball.set_position(data, random_position)

				self.observation = self.env.get_observation_space(data)
				self.done = False
				self.logger.write("Initial player position: {0}, ball position: {1}", self.env.player.get_position(data), self.env.ball.get_position(data))

		def controller(self, model, data):
				action = self.env.player.ai.choose_action(self.observation)
				new_observation, reward, self.done, info = self.env.step(data, action)
				self.env.player.ai.remember(
						self.observation, action, reward, new_observation, self.done
				)
				self.score += reward
				self.observation = new_observation

		def run(self, episode_count):
				self.reset()

				self.init_controller(self.model, self.data)
				mj.set_mjcb_control(self.controller)

				step_counter = 0
				self.score = 0

				while not glfw.window_should_close(self.window):
						time_prev = self.data.time

						while self.data.time - time_prev < 1 / self.frames_per_second:
								step_counter += 1
								mj.mj_step(self.model, self.data)

						# Train networks for every 4 steps
						if step_counter % 4 == 0:
								self.env.player.ai.learn()

						if self.done or (self.data.time > self.sim_length):
								break

						mj.mjv_updateScene(
								self.model,
								self.data,
								self.opt,
								None,
								self.cam,
								mj.mjtCatBit.mjCAT_ALL.value,
								self.scene,
						)

						viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
						viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

						mj.mjr_render(viewport, self.scene, self.context)

						glfw.swap_buffers(self.window)
						glfw.poll_events()

				self.logger.write("Final player position: {0}, ball position: {1}", self.env.player.get_position(self.data), self.env.ball.get_position(self.data))
				self.logger.write("Episode count {0}, Score: {1}", episode_count, self.score)

		def save_models(self):
				self.env.player.ai.save_models()

		def stop(self):
				glfw.terminate()
