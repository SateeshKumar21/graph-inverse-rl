import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import augmentations
import algorithms.modules as m
from algorithms.sacv2 import SACv2


class SVEAv2(SACv2):
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		self.svea_alpha = args.svea_alpha
		self.svea_beta = args.svea_beta
		self.svea_augmentation = args.svea_augmentation
		self.naive = args.naive

	def augment(self, obs):
		if self.svea_augmentation == 'colorjitter':
			return augmentations.random_color_jitter(obs.clone())
		elif self.svea_augmentation == 'affine+colorjitter':
			return augmentations.random_color_jitter(augmentations.random_affine(obs.clone()))
		elif self.svea_augmentation == 'noise':
			return augmentations.random_noise(obs.clone())
		elif self.svea_augmentation == 'affine+noise':
			return augmentations.random_noise(augmentations.random_affine(obs.clone()))
		elif self.svea_augmentation == 'conv':
			return augmentations.random_conv(obs.clone())
		elif self.svea_augmentation == 'affine+conv':
			return augmentations.random_conv(augmentations.random_affine(obs.clone()))
		elif self.svea_augmentation == 'overlay':
			return augmentations.random_overlay(obs.clone())
		elif self.svea_augmentation == 'affine+overlay':
			return augmentations.random_overlay(augmentations.random_affine(obs.clone()))
		elif self.svea_augmentation == 'none':
			return obs
		else:
			raise NotImplementedError(f'Unsupported augmentation: {self.svea_augmentation}')

	def update_critic(self, obs, action, reward, next_obs, L=None, step=None):
		with torch.no_grad():
			_, policy_action, log_pi, _ = self.actor(next_obs)
			target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
			target_V = torch.min(target_Q1,
								 target_Q2) - self.alpha.detach() * log_pi
			target_Q = reward + (self.discount * target_V)

		if self.svea_augmentation != 'none' and not self.naive:
			action = utils.cat(action, action)
			target_Q = utils.cat(target_Q, target_Q)

		current_Q1, current_Q2 = self.critic(obs, action)
		critic_loss = (self.svea_alpha + self.svea_beta) * \
			(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
		if L is not None:
			L.log('train_critic/loss', critic_loss, step)

		self.critic_optimizer.zero_grad(set_to_none=True)
		critic_loss.backward()
		self.critic_optimizer.step()

	def update(self, replay_buffer, L, step):
		if step % self.update_freq != 0:
			return

		obs, action, reward, next_obs = replay_buffer.sample()
		obs = self.aug(obs) # random shift
		if self.svea_augmentation != 'none':
			if self.naive:
				obs = self.augment(obs) # naively apply strong augmentation
			else:
				obs = utils.cat(obs, self.augment(obs)) # strong augmentation

		if self.multiview:
			obs = self.encoder(obs[:,:3,:,:], obs[:,3:6,:,:])
		else:
			obs = self.encoder(obs)

		if self.svea_augmentation != 'none' and not self.naive:
			obs_unaug = obs[:obs.size(0)//2] # unaugmented observations
		else:
			obs_unaug = obs

		with torch.no_grad():
			next_obs = self.aug(next_obs)
			if self.svea_augmentation != 'none' and self.naive:
				next_obs = self.augment(next_obs) # naively apply strong augmentation
			if self.multiview:
				next_obs = self.encoder(next_obs[:,:3,:,:], next_obs[:,3:6,:,:])
			else:
				next_obs = self.encoder(next_obs)

		self.update_critic(obs, action, reward, next_obs, L, step)
		self.update_actor_and_alpha(obs_unaug.detach(), L, step)
		utils.soft_update_params(self.critic, self.critic_target, self.tau)
