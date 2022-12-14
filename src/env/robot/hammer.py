import numpy as np
import os
from gym import utils
from env.robot.base import BaseEnv, get_full_asset_path

'''
Not completely updated, consider looking at hammer_all task for latest update
'''

class HammerEnv(BaseEnv, utils.EzPickle):
	def __init__(self, xml_path, cameras, n_substeps=20, observation_type='image', reward_type='dense', image_size=84, use_xyz=False, render=False):
		BaseEnv.__init__(self,
			get_full_asset_path(xml_path),
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			reset_free=False,
			cameras=cameras,
			render=render,
			use_xyz=use_xyz,
			has_object=True
		)
		self.state_dim = (26,) if self.use_xyz else (20,)
		utils.EzPickle.__init__(self)

	def compute_reward(self, achieved_goal, goal, info):
		eef_pos = self.sim.data.get_site_xpos('tool').copy()
		object_pos = self.sim.data.get_site_xpos('nail_target').copy()
		gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint').copy()
		goal_pos = goal.copy()
		d_eef_obj = self.goal_distance(eef_pos, object_pos, self.use_xyz)
		d_eef_obj_xy = self.goal_distance(eef_pos, object_pos, use_xyz  = False)
		#d_obj_goal_xy = self.goal_distance(object_pos, goal_pos, use_xyz=False)
		d_obj_goal_z = np.abs(object_pos[2] - goal_pos[2])
		eef_z = eef_pos[2] - self.center_of_table.copy()[2] - self.default_z_offset
		obj_z = object_pos[2] - self.center_of_table.copy()[2] - self.default_z_offset

		reward = -0.1*np.square(self._pos_ctrl_magnitude) # action penalty
		# Get end effector close to the nail
		reward += -2 * d_eef_obj

		if d_obj_goal_z<=self.distance_threshold:
			reward +=  -2 * d_obj_goal_z

		return reward

	def _get_state_obs(self):
		cot_pos = self.center_of_table.copy()
		dt = self.sim.nsubsteps * self.sim.model.opt.timestep

		eef_pos = self.sim.data.get_site_xpos('tool') #- cot_pos
		eef_velp = self.sim.data.get_site_xvelp('tool') * dt
		goal_pos = self.goal# - cot_pos
		gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint')

		obj_pos = self.sim.data.get_site_xpos('nail_target') #- cot_pos
		obj_rot = self.sim.data.get_joint_qpos('nail_board:joint')[-4:]
		obj_velp = self.sim.data.get_site_xvelp('nail_target') * dt
		obj_velr = self.sim.data.get_site_xvelr('nail_target') * dt

		if not self.use_xyz:
			eef_pos = eef_pos[:2]
			eef_velp = eef_velp[:2]
			goal_pos = goal_pos[:2]
			obj_pos = obj_pos[:2]
			obj_velp = obj_velp[:2]
			obj_velr = obj_velr[:2]

		values = np.array([
			self.goal_distance(eef_pos, goal_pos, self.use_xyz),
			self.goal_distance(obj_pos, goal_pos, self.use_xyz),
			self.goal_distance(eef_pos, obj_pos, self.use_xyz),
			gripper_angle
		])

		return np.concatenate([
			eef_pos, eef_velp, goal_pos, obj_pos, obj_rot, obj_velp, obj_velr, values
		], axis=0)

	def _reset_sim(self):

		return BaseEnv._reset_sim(self)

	def _get_achieved_goal(self):
		return np.squeeze(self.sim.data.get_site_xpos('nail_target').copy())

	def _sample_object_pos(self):
		return None

	def _sample_goal(self):
		# Goal is the peg goal site position
		object_qpos = self.sim.data.get_joint_qpos('nail_board:joint')
		sampled = object_qpos[:3].copy()
		object_quat = object_qpos[-4:]
		sampled[0] += self.np_random.uniform(-0.05, 0.05, size=1)
		sampled[1] += self.np_random.uniform(-0.1, 0.1, size=1)
		object_qpos[:3] = sampled[:3].copy()
		object_qpos[-4:] = object_quat
		self.sim.data.set_joint_qpos('nail_board:joint', object_qpos)

		peg_site_xpos = self.sim.data.get_site_xpos('nail_goal')
		goal = peg_site_xpos.copy()
		
		return BaseEnv._sample_goal(self, goal)

	def _sample_initial_pos(self):
		gripper_target = self.center_of_table.copy() - np.array([0.3, 0, 0])
		gripper_target[0] += self.np_random.uniform(-0.15, -0.05, size=1)
		gripper_target[1] += self.np_random.uniform(-0.05, 0.05, size=1)
		gripper_target[2] += self.default_z_offset
		if self.use_xyz:
			gripper_target[2] += self.np_random.uniform(0, 0.1, size=1)
		BaseEnv._sample_initial_pos(self, gripper_target)
