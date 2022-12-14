import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import re
import utils
import algorithms.modules as m
import cv2

class TruncatedNormal(torch.distributions.Normal):
	def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
		super().__init__(loc, scale, validate_args=False)
		self.low = low
		self.high = high
		self.eps = eps

	def _clamp(self, x):
		clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
		x = x - x.detach() + clamped_x.detach()
		return x

	def sample(self, clip=None, sample_shape=torch.Size()):
		shape = self._extended_shape(sample_shape)
		eps = torch.distributions.utils._standard_normal(shape,
							   dtype=self.loc.dtype,
							   device=self.loc.device)
		eps *= self.scale
		if clip is not None:
			eps = torch.clamp(eps, -clip, clip)
		x = self.loc + eps
		return self._clamp(x)


def schedule(schdl, train_steps, step):
	try:
		return float(schdl)
	except ValueError:
		match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
		if match:
			init, final, duration = [float(g) for g in match.groups()]
			duration *= train_steps
			mix = np.clip(step / duration, 0.0, 1.0)
			return (1.0 - mix) * init + mix * final
	raise NotImplementedError(schdl)


class Actor(nn.Module):
	def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
		super().__init__()
		self.layers = nn.Sequential(nn.Linear(repr_dim, feature_dim),
									nn.LayerNorm(feature_dim), nn.Tanh(),
									nn.Linear(feature_dim, hidden_dim),
									nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, hidden_dim),
									nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, action_shape[0]), nn.Tanh())
		self.apply(m.orthogonal_init)

	def forward(self, obs):
		return self.layers(obs)


class Critic(nn.Module):
	def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
		super().__init__()
		self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
								   nn.LayerNorm(feature_dim), nn.Tanh())
		self.Q1 = nn.Sequential(
			nn.Linear(feature_dim + action_shape[0], hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
		self.Q2 = nn.Sequential(
			nn.Linear(feature_dim + action_shape[0], hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
		self.apply(m.orthogonal_init)

	def forward(self, obs, action):
		h = torch.cat([self.trunk(obs), action], dim=-1)
		return self.Q1(h), self.Q2(h)


class DrQv2(object): # DDPG
	def __init__(self, obs_shape, action_shape, args):
		self.discount = args.discount
		self.update_freq = args.update_freq
		self.tau = args.tau
		self.num_expl_steps = args.num_expl_steps
		self.train_steps = args.train_steps
		self.std_schedule = args.std_schedule
		self.std_clip = args.std_clip
		assert not args.from_state and not args.use_vit, 'not supported yet'
		self.attention = bool(args.attention)
		self.concatenate = bool(args.concat)
		self.context1 = bool(args.context1)
		self.context2 = bool(args.context2)

		if args.cameras==2:
			self.multiview = True
		else:
			self.multiview = False

		if self.multiview:
			obs_shape = list(obs_shape)
			obs_shape[0] = 3
			shared_1 = m.SharedCNN(obs_shape, args.num_shared_layers, args.num_filters, args.mean_zero)
			shared_2 = m.SharedCNN(obs_shape, args.num_shared_layers, args.num_filters, args.mean_zero)

			integrator = m.Integrator(shared_1.out_shape, shared_2.out_shape, args.num_filters, concatenate=self.concatenate) # Change channel dimensions of concatenated features

			assert shared_1.out_shape==shared_2.out_shape, 'Image features must be the same'
			
			
			if self.attention:
				attention1 = None
				attention2 = None
				if self.context1 or self.context2:
					head = m.HeadCNN(shared_1.out_shape, args.num_head_layers, args.num_filters, flatten=True)
					
					if self.context1:
						attention1 = m.AttentionBlock(dim=shared_1.out_shape, contextualReasoning=self.context1)
					if self.context2:
						attention2 = m.AttentionBlock(dim=shared_1.out_shape, contextualReasoning=self.context2)

					self.encoder = m.MultiViewEncoder(
						shared_1,
						shared_2,
						integrator,
						head,
						m.Identity(out_dim=head.out_shape[0]),
						attention1,
						attention2,
						concatenate=self.concatenate,
						contextualReasoning1=self.context1,
						contextualReasoning2=self.context2
					).cuda()
				else:
					head = m.HeadCNN(shared_1.out_shape, args.num_head_layers, args.num_filters, flatten=False)
					attention1 = m.AttentionBlock(dim=head.out_shape, contextualReasoning=False)

					self.encoder = m.MultiViewEncoder(
						shared_1,
						shared_2,
						integrator,
						head,
						m.Identity(out_dim=attention1.out_shape[0]),
						attention1,
						attention2,
						concatenate=self.concatenate,
						contextualReasoning1=False,
						contextualReasoning2=False
					).cuda()
			else:
				head = m.HeadCNN(shared_1.out_shape, args.num_head_layers, args.num_filters)
				self.encoder = m.MultiViewEncoder(
					shared_1,
					shared_2,
					integrator,
					head,
					m.Identity(out_dim=head.out_shape[0]),
					concatenate=self.concatenate
				).cuda()
		else:
			shared = m.SharedCNN(obs_shape, args.num_shared_layers, args.num_filters, args.mean_zero)
			head = m.HeadCNN(shared.out_shape, args.num_head_layers, args.num_filters)
			self.encoder = m.Encoder(
				shared,
				head,
				m.Identity(out_dim=head.out_shape[0])
			).cuda()


		self.actor = Actor(self.encoder.out_dim, action_shape, args.projection_dim, args.hidden_dim).cuda()
		self.critic = Critic(self.encoder.out_dim, action_shape, args.projection_dim, args.hidden_dim).cuda()
		self.critic_target = Critic(self.encoder.out_dim, action_shape, args.projection_dim, args.hidden_dim).cuda()
		self.critic_target.load_state_dict(self.critic.state_dict())

		self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
		self.critic_optim = torch.optim.Adam(itertools.chain(self.encoder.parameters(), self.critic.parameters()), lr=args.lr)

		self.aug = m.RandomShiftsAug(pad=4)
		self.train()
		print('Encoder:', utils.count_parameters(self.encoder))
		print('Actor:', utils.count_parameters(self.actor))
		print('Critic:', utils.count_parameters(self.critic))

	def train(self, training=True):
		self.training = training
		for p in [self.encoder, self.actor, self.critic, self.critic_target]:
			p.train(training)

	def eval(self):
		self.train(False)
		
	def _obs_to_input(self, obs):
		if isinstance(obs, utils.LazyFrames):
			_obs = np.array(obs)
		else:
			_obs = obs
		_obs = torch.FloatTensor(_obs).cuda()
		_obs = _obs.unsqueeze(0)
		return _obs

	def select_action(self, obs):
		_obs = self._obs_to_input(obs)
		with torch.no_grad():
			if self.multiview:
				mu = self.actor(self.encoder(_obs[:,:3,:,:], _obs[:,3:6,:,:]))
			else:
				mu = self.actor(self.encoder(_obs))
		return mu.cpu().data.numpy().flatten()

	def _add_noise(self, mu, step, clip=None):
		std = torch.ones_like(mu) * schedule(self.std_schedule, self.train_steps, step)
		dist = TruncatedNormal(mu, std)
		return dist.sample(clip=clip)

	def sample_action(self, obs, step):
		_obs = self._obs_to_input(obs)
		with torch.no_grad():
			if self.multiview:
				mu = self.actor(self.encoder(_obs[:,:3,:,:], _obs[:,3:6,:,:]))
			else:
				mu = self.actor(self.encoder(_obs))

		a = self._add_noise(mu, step)
		if step < self.num_expl_steps:
			a.uniform_(-1.0, 1.0)
		return a.cpu().numpy().flatten()



	def update_critic(self, obs, action, reward, next_obs, L=None, step=None):
		with torch.no_grad():
			mu = self.actor(next_obs)
			next_action = self._add_noise(mu, step, clip=self.std_clip)
			target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
			target_V = torch.min(target_Q1, target_Q2)
			target_Q = reward + self.discount * target_V

		Q1, Q2 = self.critic(obs, action)
		critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
		if L is not None:
			L.log('train_critic/loss', critic_loss, step)

		self.critic_optim.zero_grad(set_to_none=True)
		critic_loss.backward()
		self.critic_optim.step()

	def update_actor(self, obs, L=None, step=None):
		mu = self.actor(obs)
		action = self._add_noise(mu, step, clip=self.std_clip)
		Q1, Q2 = self.critic(obs, action)
		Q = torch.min(Q1, Q2)

		actor_loss = -Q.mean()
		if L is not None:
			L.log('train_actor/loss', actor_loss, step)

		self.actor_optim.zero_grad(set_to_none=True)
		actor_loss.backward()
		self.actor_optim.step()

	def update(self, replay_buffer, L, step):
		if step % self.update_freq != 0:
			return
		obs, action, reward, next_obs = replay_buffer.sample()
		obs = self.aug(obs)

		if self.multiview:
			obs = self.encoder(obs[:,:3,:,:], obs[:,3:6,:,:])
		else:
			obs = self.encoder(obs)

		with torch.no_grad():
			next_obs = self.aug(next_obs)
			if self.multiview:
				next_obs = self.encoder(next_obs[:,:3,:,:], next_obs[:,3:6,:,:])
			else:
				next_obs = self.encoder(next_obs)

		self.update_critic(obs, action, reward, next_obs, L, step)
		self.update_actor(obs.detach(), L, step)
		utils.soft_update_params(
			self.critic, self.critic_target, self.tau
		)
