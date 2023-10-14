import mujoco as mj

class Ball:
	def __init__(self, model, name):
		self.name = name

		self.id_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
		self.id_geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
		self.id_joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)

	def get_position(self, data):
		return data.qpos[self.id_joint * 7: self.id_joint * 7 + 3]

	def set_position(self, data, position):
		data.qpos[self.id_joint * 7: self.id_joint * 7 + 3] = position

	def get_velocity(self, data):
		return data.qvel[self.id_joint * 6: self.id_joint * 6 + 6]