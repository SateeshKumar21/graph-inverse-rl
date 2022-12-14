import numpy as np
from numpy.random import randint
import os
import gym
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from gym.wrappers import TimeLimit
from env.robot import registration
import utils
from collections import deque
from mujoco_py import modder


def make_env(
		domain_name,
		task_name,
		seed=0,
		episode_length=50,
		n_substeps=20,
		frame_stack=3,
		action_repeat=1,
		image_size=84,
		cameras=['third_person', 'first_person'],
		mode='train',
		render=False,
		observation_type='image',
		camera_dropout=0,
		action_space='xyzw',
		rand_first=False,
		use_3d=False
	):
	"""Make environment for experiments"""
	assert domain_name == 'robot', f'expected domain_name "robot", received "{domain_name}"'
	assert action_space in {'xy', 'xyz', 'xyzw'}, f'unexpected action space "{action_space}"'

	print("TYPE ", observation_type)
	registration.register_robot_envs(
		n_substeps=n_substeps,
		observation_type=observation_type,
		image_size=image_size,
		use_xyz=action_space.replace('w', '') == 'xyz')
	randomizations = {}
	env_id = 'Robot' + task_name.capitalize() + '-v0'
	env = gym.make(env_id, cameras=cameras, render=render, observation_type=observation_type)
	env.seed(seed)
	env = TimeLimit(env, max_episode_steps=episode_length)
	env = SuccessWrapper(env, any_success=True)
	
	if not use_3d:
		env = ObservationSpaceWrapper(env, observation_type=observation_type, image_size=image_size, cameras=cameras, camera_dropout=camera_dropout)
	else:
		env = ObservationSpaceWrapper3D(env, observation_type=observation_type, image_size=image_size, cameras=cameras, camera_dropout=camera_dropout)
	
	env = ActionSpaceWrapper(env, action_space=action_space)
	env = FrameStack(env, frame_stack)
	if use_3d:
		env = DomainRandomizationWrapper(env, domain_name=domain_name, randomizations=randomizations, seed=seed,
                                     camera_name=cameras, rand_first=rand_first)

	return env


class FrameStack(gym.Wrapper):
	"""Stack frames as observation"""
	def __init__(self, env, k):
		gym.Wrapper.__init__(self, env)
		self._k = k
		self._frames = deque([], maxlen=k)
		shp = env.observation_space.shape
		if len(shp) == 3:
			self.observation_space = gym.spaces.Box(
				low=0,
				high=1,
				shape=((shp[0] * k,) + shp[1:]),
				dtype=env.observation_space.dtype
			)
		else:
			self.observation_space = gym.spaces.Box(
				low=-np.inf,
				high=np.inf,
				shape=(shp[0] * k,),
				dtype=env.observation_space.dtype
			)
		self._max_episode_steps = env._max_episode_steps

	def reset(self):
		obs, state_obs = self.env.reset()
		for _ in range(self._k):
			self._frames.append(obs)
		return self._get_obs(), state_obs

	def step(self, action):
		obs, state_obs, reward, done, info = self.env.step(action)
		self._frames.append(obs)
		return self._get_obs(), state_obs, reward, done, info

	def _get_obs(self):
		assert len(self._frames) == self._k
		return utils.LazyFrames(list(self._frames))


class SuccessWrapper(gym.Wrapper):
	def __init__(self, env, any_success=True):
		gym.Wrapper.__init__(self, env)
		self._max_episode_steps = env._max_episode_steps
		self.any_success = any_success
		self.success = False

	def reset(self):
		self.success = False
		return self.env.reset()

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		if self.any_success:
			self.success = self.success or bool(info['is_success'])
		else:
			self.success = bool(info['is_success'])
		info['is_success'] = self.success
		return obs, reward, done, info


class ObservationSpaceWrapper(gym.Wrapper):
	def __init__(self, env, observation_type, image_size, cameras, camera_dropout):
		#assert observation_type in {'state', 'image'}, 'observation type must be one of \{state, image\}'
		gym.Wrapper.__init__(self, env)
		self._max_episode_steps = env._max_episode_steps
		self.observation_type = observation_type
		self.image_size = image_size
		self.cameras = cameras
		self.num_cams = len(self.cameras)
		self.camera_dropout = camera_dropout # [0,1,2,3] 0: None, 1: TP, 2: FP, 3:Random

		if self.observation_type in {'image', 'state+image', 'statefull+image'}:
			self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3*self.num_cams, image_size, image_size), dtype=np.uint8)

		elif self.observation_type == 'state':
			self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=env.unwrapped.state_dim, dtype=np.float32)

	def reset(self):
		obs = self.env.reset()
		return self._get_obs(obs), obs['state'] if 'state' in obs else None

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		return self._get_obs(obs), obs['state'] if 'state' in obs else None, reward, done, info

	def _get_obs(self, obs_dict):
		if self.camera_dropout==3:
			leave_out = 1 + np.random.randint(0, 3) # Chose a number b/w 1,2,3 at random
		else:
			leave_out = self.camera_dropout

		if self.observation_type in {'image', 'state+image', 'statefull+image'}:
			if self.num_cams == 1:
				return obs_dict['observation'][0].transpose(2, 0, 1)
			obs = np.empty((3*self.num_cams, self.image_size, self.image_size), dtype=obs_dict['observation'][0].dtype)
			for ob in range(obs_dict['observation'].shape[0]):
				if leave_out==(ob+1):
					continue
				else:
					obs[3*ob:3*(ob+1)] = obs_dict['observation'][ob].transpose(2, 0, 1)


		elif self.observation_type == 'state':
			obs = obs_dict['observation']
		return obs


class ObservationSpaceWrapper3D(gym.Wrapper):
    def __init__(self, env, observation_type, image_size, cameras, camera_dropout):
        # assert observation_type in {'state', 'image'}, 'observation type must be one of \{state, image\}'
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps
        self.observation_type = observation_type
        self.image_size = image_size
        self.cameras = cameras
        self.num_cams = len(self.cameras)
        self.camera_dropout = camera_dropout  # [0,1,2,3] 0: None, 1: TP, 2: FP, 3:Random

        if self.observation_type == 'image':
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3 * 2, image_size, image_size),
                                                    dtype=np.uint8)

        elif self.observation_type == 'state':
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=env.unwrapped.state_dim,
                                                    dtype=np.float32)

    def reset(self):
        return self._get_obs(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs_dict):
        obs = obs_dict['observation']

        if self.observation_type == 'image':
            output = np.empty((3 * obs.shape[0], self.image_size, self.image_size), dtype=obs.dtype)
            for i in range(obs.shape[0]):
                output[3 * i: 3 * (i + 1)] = obs[i].transpose(2, 0, 1)
        elif self.observation_type == 'state':
            output = obs_dict['observation']
        return output


class ActionSpaceWrapper(gym.Wrapper):
	def __init__(self, env, action_space):
		assert action_space in {'xy', 'xyz', 'xyzw'}, 'task must be one of {xy, xyz, xyzw}'
		gym.Wrapper.__init__(self, env)
		self._max_episode_steps = env._max_episode_steps
		self.action_space_dims = action_space
		self.use_xyz = 'xyz' in action_space
		self.use_gripper = 'w' in action_space
		self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2+self.use_xyz+self.use_gripper,), dtype=np.float32)
	
	def step(self, action):
		assert action.shape == self.action_space.shape, 'action shape must match action space'
		action = np.array([action[0], action[1], action[2] if self.use_xyz else 0, action[3] if self.use_gripper else 1], dtype=np.float32)
		return self.env.step(action)

class DomainRandomizationWrapper(gym.Wrapper):
	def __init__(self, env, domain_name, camera_name, rand_first=0, randomizations=None, seed=None):
		# assert domain_name in {'reach', 'push', 'cloth'}, \
		#	'domain randomization only implemented for reach, push, cloth domains'
		gym.Wrapper.__init__(self, env)
		self._max_episode_steps = env._max_episode_steps
		self.domain_name = domain_name
		self.valid_randomizations = {'camera', 'light', 'material', 'skybox', 'texture', 'brightness'}
		if randomizations is None:
			randomizations = {}
		assert isinstance(randomizations, (dict, set)), \
			'randomizations must be one of {dict, set, "all", None}'
		for randomization in randomizations:
			assert randomization in self.valid_randomizations, \
				f'received invalid randomization: "{randomization}"'
		self.randomizations = randomizations
		self.sim = self.env.unwrapped.sim
		self.random_state = np.random.RandomState(seed)
		self.camera_name = "camera_" + camera_name
		self.cam_modder = modder.CameraModder(self.sim, random_state=self.random_state)
		self.light_name = 'light0'
		self.light_modder = modder.LightModder(self.sim, random_state=self.random_state)
		self.geom_names = ['tablegeom0', 'floorgeom0']
		self.material_modder = modder.MaterialModder(self.sim, random_state=self.random_state)
		self.texture_modder = modder.TextureModder(self.sim, random_state=self.random_state)
		self.brightness_std = 0
		self.rand_first = rand_first

	def reset(self):
		if 'texture' in self.randomizations:
			self._randomize_texture()
		if 'camera' in self.randomizations:
			self._randomize_camera()
		if 'light' in self.randomizations:
			self._randomize_light()
		if 'material' in self.randomizations:
			self._randomize_material()
		if 'skybox' in self.randomizations and not 'texture' in self.randomizations:
			self._randomize_skybox()
		if 'brightness' in self.randomizations:
			self._randomize_brightness()

		return self._modify_obs(self.env.reset())

	def step(self, action):
		# self._randomize_texture()
		# if "camera" in self.randomizations:
		self._randomize_camera()
		obs, reward, done, info = self.env.step(action)
		return self._modify_obs(obs), reward, done, info

	def _modify_obs(self, obs):
		"""if len(obs.shape) > 1:
			return (np.clip(obs/255. + np.ones_like(obs) * self.brightness_std, 0, 1)*255).astype(np.uint8)"""
		return obs

	def _randomize_texture(self):
		for name in self.geom_names:
			self.texture_modder.whiten_materials()
			self.texture_modder.set_checker(name, (255, 0, 0), (0, 0, 0))
			self.texture_modder.rand_all(name)
		self.texture_modder.rand_gradient('skybox')

	# reset object0
	# geom_id = self.sim.model.geom_name2id('object0')
	# mat_id = self.sim.model.geom_matid[geom_id]
	# self.sim.model.mat_rgba[mat_id] = np.array([0.82, 0.67, 0.22, 1])

	def _randomize_camera(self):
		# self.cam_modder.set_fovy(self.camera_name, self.random_state.randint(45, 50))
		# self.cam_modder.set_pos(self.camera_name, self.pos + self._uniform(self.pos, [0, -0.25, 0], [0, 0.25, 0]))
		theta = np.random.rand() * np.pi - np.pi/2  #(3 * np.pi / 2) - (3 * np.pi / 4)
		phi = np.random.rand() * np.pi/2
		cos_theta = np.cos(theta)
		sin_theta = np.sin(theta)

		cos_phi = np.cos(phi)
		sin_phi = np.sin(phi)

		pos = self.cam_modder.get_pos(self.camera_name)

		# new_x = 1.25 + 0.7 * cos_theta
		new_x = 1.5 + 0.5 * cos_phi * cos_theta
		new_y = 0.3 + 0.3 * cos_phi * sin_theta
		new_z = 1.4 + 0.1 * sin_phi

		# new_x = 1.655 + 0.445 * cos_theta
		# new_y = 0.3 + 0.6 * sin_theta

		pos[0] = new_x
		pos[1] = new_y
		pos[2] = new_z

		self.cam_modder.set_pos(self.camera_name, pos)

		if self.rand_first:
			theta = np.random.rand() * np.pi / 6 - np.pi / 12  # (3 * np.pi / 2) - (3 * np.pi / 4)
			phi = np.random.rand() * np.pi / 12
			cos_theta = np.cos(theta)
			sin_theta = np.sin(theta)

			cos_phi = np.cos(phi)
			sin_phi = np.sin(phi)

			pos = self.cam_modder.get_pos("camera_front")

			# new_x = 1.25 + 0.7 * cos_theta
			new_x = 1.5 + 0.5 * cos_phi * cos_theta
			new_y = 0.3 + 0.3 * cos_phi * sin_theta
			new_z = 1.4 + 0.1 * sin_phi

			# new_x = 1.655 + 0.445 * cos_theta
			# new_y = 0.3 + 0.6 * sin_theta

			pos[0] = new_x
			pos[1] = new_y
			pos[2] = new_z

			self.cam_modder.set_pos("camera_front", pos)


	def _randomize_brightness(self):
		self.brightness_std = self._add_noise(0, 0.05)

	def _randomize_light(self):
		self.light_modder.set_ambient(self.light_name, self._uniform([0.2, 0.2, 0.2]))
		self.light_modder.set_diffuse(self.light_name, self._uniform([0.8, 0.8, 0.8]))
		self.light_modder.set_specular(self.light_name, self._uniform([0.3, 0.3, 0.3]))
		self.light_modder.set_pos(self.light_name, self._add_noise([0, 0, 4], 0.5))
		self.light_modder.set_dir(self.light_name, self._add_noise([0, 0, -1], 0.25))
		self.light_modder.set_castshadow(self.light_name, self.random_state.randint(0, 2))

	def _randomize_material(self):
		for name in self.geom_names:
			self.material_modder.rand_all(name)

	def _randomize_skybox(self):
		self.texture_modder.rand_gradient('skybox')
		geom_id = self.sim.model.geom_name2id('floorgeom0')
		mat_id = self.sim.model.geom_matid[geom_id]
		self.sim.model.mat_rgba[mat_id] = np.clip(self._add_noise([0.2, 0.15, 0.1, 1], [0.1, 0.2, 0.2, 0]), 0, 1)

	def _uniform(self, default, low=0.0, high=1.0):
		if isinstance(default, list):
			default = np.array(default)
		if isinstance(low, list):
			assert len(low) == len(default), 'low and default must be same length'
			low = np.array(low)
		if isinstance(high, list):
			assert len(high) == len(default), 'high and default must be same length'
			high = np.array(high)
		return np.random.uniform(low=low, high=high, size=len(default))

	def _add_noise(self, default, std):
		if isinstance(default, list):
			default = np.array(default)
		elif isinstance(default, (float, int)):
			default = np.array([default], dtype=np.float32)
		if isinstance(std, list):
			assert len(std) == len(default), 'std and default must be same length'
			std = np.array(std)
		return default + std * self.random_state.randn(len(default))
