import numpy as np
import os
from gym import utils
from env.robot.base import BaseEnv, get_full_asset_path


class GolfXYEnv(BaseEnv, utils.EzPickle):
	def __init__(self, xml_path, n_substeps=20, observation_type='image', reward_type='dense', image_size=84):
		BaseEnv.__init__(self,
			get_full_asset_path(xml_path),
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			reset_free=False,
			distance_threshold=0.035,
			use_xyz=False,
			action_scale=0.1,
			has_object=True
		)
		utils.EzPickle.__init__(self)
		self.default_z_offset = 0.2

	def compute_reward(self, achieved_goal, goal, info):
		d = self.goal_distance(achieved_goal, goal, self.use_xyz)
		if self.reward_type == 'sparse':
			return -(d > self.distance_threshold).astype(np.float32)
		else:
			penalty = -self._pos_ctrl_magnitude * self.action_penalty
			if self.reward_bonus and d <= self.distance_threshold:
				return np.around(1-d, 4) + penalty
			return np.around(-d, 4) + penalty

	def _step_callback(self):
		pass

	def _get_achieved_goal(self):
		return np.squeeze(self.sim.data.get_site_xpos('object0').copy())

	def _limit_gripper(self, gripper_pos, pos_ctrl):
		if gripper_pos[0] > 1.4:
			pos_ctrl[0] = min(pos_ctrl[0], 0)
		if gripper_pos[0] < 1.2:
			pos_ctrl[0] = max(pos_ctrl[0], 0)
		if gripper_pos[1] > 0.06:
			pos_ctrl[1] = min(pos_ctrl[1], 0)
		if gripper_pos[1] < -0.2:
			pos_ctrl[1] = max(pos_ctrl[1], 0)
		return pos_ctrl

	def _sample_object_pos(self):
		object_xpos = self.center_of_table.copy()
		object_xpos[0] += self.np_random.uniform(-0.04, 0.04, size=1)
		object_xpos[1] += self.np_random.uniform(-0.04, 0.04, size=1)
		object_xpos[2] += 0.025
		object_qpos = self.sim.data.get_joint_qpos('object0:joint')
		object_quat = object_qpos[-4:]
		object_quat[0] = self.np_random.uniform(-1, 1, size=1)
		object_quat[3] = self.np_random.uniform(-1, 1, size=1)

		assert object_qpos.shape == (7,)
		object_qpos[:3] = object_xpos[:3]
		object_qpos[-4:] = object_quat
		self.sim.data.set_joint_qpos('object0:joint', object_qpos)

	def _sample_goal(self):
		goal = np.array([1.3, 0.4, 0.7])
		goal[1] += self.np_random.uniform(-0.05, 0.025, size=1)
		return BaseEnv._sample_goal(self, goal)

	def _sample_initial_pos(self):
		gripper_target = self.center_of_table.copy()
		gripper_target[0] += self.np_random.uniform(-0.06, 0.06, size=1)
		gripper_target[1] += self.np_random.uniform(-0.2, -0.15, size=1)
		gripper_target[2] += self.default_z_offset
		BaseEnv._sample_initial_pos(self, gripper_target)
