import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import utils
import algorithms.modules as m
import algorithms.modules_3d as m3d

from algorithms.rot_utils import euler2mat


class SACv2_3D(object):
    def __init__(self, obs_shape, in_shape, action_shape, args):
        self.discount = args.discount
        self.update_freq = args.update_freq
        self.tau = args.tau
        assert not args.from_state and not args.use_vit, 'not supported yet'

        self.train_rl = args.train_rl
        self.train_3d = args.train_3d
        self.prop_to_3d = args.prop_to_3d
        self.log_3d_imgs = args.log_3d_imgs
        self.huber = args.huber
        self.bsize_3d = args.bsize_3d
        self.update_3d_freq = args.update_3d_freq
        self.only_2_recon = args.only_2_recon
        self.use_vae = args.use_vae

        project = True if args.use_latent else False
        project_conv = True if args.project_conv == 1 else False

        shared = m.SharedCNNv2(obs_shape, in_shape, args.num_shared_layers, args.num_filters, project, project_conv)
        head = m.HeadCNN(shared.out_shape, args.num_head_layers, args.num_filters)
        self.encoder_rl = m.Encoder(
            shared,
            head,
            m.Identity(out_dim=head.out_shape[0])
        ).cuda()

        self.actor = m.EfficientActor(self.encoder_rl.out_dim, args.projection_dim, action_shape, args.hidden_dim,
                                      args.actor_log_std_min, args.actor_log_std_max).cuda()
        self.critic = m.EfficientCritic(self.encoder_rl.out_dim, args.projection_dim, action_shape, args.hidden_dim).cuda()
        self.critic_target = m.EfficientCritic(self.encoder_rl.out_dim, args.projection_dim, action_shape,
                                               args.hidden_dim).cuda()
        self.critic_target.load_state_dict(self.critic.state_dict())

        """
        3D Networks
        """
        self.encoder_3d = m3d.Encoder_3d(args).cuda()
        self.decoder_3d = m3d.Decoder_3d(args).cuda()
        self.rotate_3d = m3d.Rotate_3d(args).cuda()
        self.pose_3d = m3d.Posenet_3d().cuda()

        self.log_alpha = torch.tensor(np.log(args.init_temperature)).cuda()
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(action_shape)

        """
        RL Optimizers
        """
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_optimizer = torch.optim.Adam(itertools.chain(self.encoder_3d.parameters(),
                                                                 self.encoder_rl.parameters(),
                                                                 self.critic.parameters()),
                                                 lr=args.lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr, betas=(args.alpha_beta, 0.999))

        """
        3D Optimizers
        """

        self.recon3d_optimizer = torch.optim.Adam(itertools.chain(self.encoder_3d.parameters(),
                                                                  self.rotate_3d.parameters(),
                                                                  self.decoder_3d.parameters()), lr=args.lr_3dc)
        self.pose3d_optimizer = torch.optim.Adam(self.pose_3d.parameters(), lr=args.lr_3dp)

        self.aug = m.RandomShiftsAug(pad=4)
        self.train()

        print("\n3D Encoder:", utils.count_parameters(self.encoder_3d))
        print('RL Encoder:', utils.count_parameters(self.encoder_rl))
        print('\nActor:', utils.count_parameters(self.actor))
        print('Critic:', utils.count_parameters(self.critic))
        print("\n3D Decoder: ", utils.count_parameters(self.decoder_3d))
        print("3D RotNet: ", utils.count_parameters(self.rotate_3d))
        print("3D PoseNet: ", utils.count_parameters(self.pose_3d))


    def train(self, training=True):
        self.training = training
        for p in [self.encoder_rl, self.actor, self.critic, self.critic_target]:
            p.train(training)
        for p in [self.encoder_3d, self.decoder_3d, self.rotate_3d, self.pose_3d]:
            p.train(training)


    def eval(self):
        self.train(False)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _obs_to_input(self, obs):
        if isinstance(obs, utils.LazyFrames):
            _obs = np.array(obs)
        else:
            _obs = obs
        _obs = torch.FloatTensor(_obs).cuda()
        _obs = _obs.unsqueeze(0)
        return _obs

    def select_action(self, obs):
        obs = obs[:3]
        _obs = self._obs_to_input(obs).div(255)
        with torch.no_grad():
            _obs, _ = self.encoder_3d(_obs)
            mu, _, _, _ = self.actor(self.encoder_rl(_obs), compute_pi=False, compute_log_pi=False)
        return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs, step=None):
        obs = obs[:3]
        _obs = self._obs_to_input(obs).div(255)
        with torch.no_grad():
            _obs, _ = self.encoder_3d(_obs)
            mu, pi, _, _ = self.actor(self.encoder_rl(_obs), compute_log_pi=False)
        return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, L=None, writer=None, step=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (self.discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        if L is not None:
            L.log('train_critic/loss', critic_loss, step)
        if writer is not None:
            writer.add_scalar("Critic Loss (Training)", critic_loss, step)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, L=None, writer=None, step=None, update_alpha=True):
        _, pi, log_pi, log_std = self.actor(obs)
        Q1, Q2 = self.critic(obs, pi)
        Q = torch.min(Q1, Q2)
        actor_loss = (self.alpha.detach() * log_pi - Q).mean()
        if L is not None:
            L.log('train_actor/loss', actor_loss, step)
        if writer is not None:
            writer.add_scalar("Actor Loss (Training)", actor_loss, step)

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        if update_alpha:
            self.log_alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

            if L is not None:
                L.log('train_alpha/loss', alpha_loss, step)
                L.log('train_alpha/value', self.alpha, step)

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def fwd_3d(self, imgs, writer, step, pose=False, log=False):
        """

        :param imgs:
        :param writer: Tensorboard
        :param step:
        :param log: Train pose or encoder(also log in this case)
        :return:
        """
        b, t, c, h, w = imgs.size()
        # if log:
        _, latent_3d = self.encoder_3d(imgs[:, 0])
        # else:
        #    with torch.no_grad():
        #       _, latent_3d = self.encoder_3d(imgs[:, 0])
        _, C, H, W, D = latent_3d.size()

        # Duplicate the representation for each view
        object_code_t = latent_3d.unsqueeze(1).repeat(1, t, 1, 1, 1, 1).view(b * t, C, H, W, D)

        # Get Poses
        imgs_ref = imgs[:, 0:1].repeat(1, t - 1, 1, 1, 1)
        imgs_pair = torch.cat([imgs_ref, imgs[:, 1:]], dim=2)  # b x t-1 x 6 x h x w
        pair_tensor = imgs_pair.view(b * (t - 1), c * 2, h, w)

        # if log:
        #   with torch.no_grad():
        #        traj = self.pose_3d(pair_tensor)  # b*t-1 x 6
        # else:
        traj_mean, traj_var = self.pose_3d(pair_tensor)
        traj_stdev = torch.exp(0.5 * traj_var)
        eps = torch.randn_like(traj_stdev)
        if self.use_vae:
            traj = traj_mean + eps * traj_stdev
        else:
            traj = traj_mean
        poses = torch.cat([torch.zeros(b, 1, 6).cuda(), traj.view(b, t - 1, 6)], dim=1).view(b * t, 6)

        theta = euler2mat(poses, scaling=False, translation=True)

        # if log:
        rot_codes = self.rotate_3d(object_code_t, theta)
        # else:
        #    with torch.no_grad():
        #        rot_codes = self.rotate_3d(object_code_t, theta)

        # Decode the representation to get back image.
        # if log:
        output = self.decoder_3d(rot_codes)
        # else:
        #    with torch.no_grad():
        #        output = self.decoder_3d(rot_codes)
        output = F.interpolate(output, (h, w), mode='bilinear')  # T*B x 3 x H x W
        img_tensor = imgs.view(b * t, c, h, w)

        # L2 Loss
        output_loss = output
        if self.only_2_recon:
            output_loss = output.view(b, t, c, h, w)[:, 1, ...]
            img_tensor = imgs[:, 1, ...]
        if not self.huber:
            loss_3d = F.mse_loss(output_loss, img_tensor)
        else:
            loss_3d = F.smooth_l1_loss(output_loss, img_tensor)

        if self.use_vae and pose:
            kld_loss = torch.mean(-0.5 * torch.sum(1 + traj_var - traj_mean ** 2 - traj_var.exp(), dim=1), dim=0)

        if self.use_vae and pose and writer is not None:
            writer.add_scalar("Loss KL Pose", kld_loss, step)
        if log and writer is not None:
            writer.add_scalar("Loss 3D Recon", loss_3d, step)
            if step % self.log_3d_imgs == 0:
                writer.add_images(f'input images (Training)', imgs[0], step)
                writer.add_images(f'reconstruction images (Training)',
                                  torch.clamp(output.view(b, t, c, h, w)[0], 0, 1), step)

                writer.add_video(f'input videos (Training)', imgs, step)
                writer.add_video(f'reconstruction videos (Training)',
                                 torch.clamp(output, 0, 1).view(b, t, c, h, w), step)

        if self.use_vae and pose:
            return loss_3d + kld_loss
        return loss_3d

    def update_3d_recon(self, imgs, L=None, writer=None, step=None):
        """
        Uppdate 3D Networks
        :param imgs: b x t x c x h x w
        :param L: Logger
        :param writer: Tensorboard SummaryWriter
        :param step: Train Step
        :return:
        """
        # imgs = imgs.div(255)
        self.recon3d_optimizer.zero_grad(set_to_none=True)

        loss = self.fwd_3d(imgs, writer, step, log=True)
        if L is not None:
            L.log("train_3d/loss", loss, step)
        loss.backward()

        self.recon3d_optimizer.step()

        self.pose3d_optimizer.zero_grad(set_to_none=True)

        loss = self.fwd_3d(imgs, writer, step, pose=True, log=False)
        loss.backward()

        self.pose3d_optimizer.step()

    def gen_interpolate(self, imgs, writer=None, step=None):
        with torch.no_grad():
            b, t, c, h, w = imgs.size()
            _, latent_3d = self.encoder_3d(imgs[:, 0])
            _, C, H, W, D = latent_3d.size()

            a = torch.tensor(np.arange(0., 1.1, 0.1)).to(latent_3d.device).unsqueeze(0).repeat(b, 1).view(b, -1)
            object_code_t = latent_3d.unsqueeze(1).repeat(1, a.size(1), 1, 1, 1, 1).view(b * (a.size(1)), C, H, W, D)

            imgs_ref = imgs[:, 0:1].repeat(1, t - 1, 1, 1, 1)
            imgs_pair = torch.cat([imgs_ref, imgs[:, 1:]], dim=2)  # b x t-1 x 6 x h x w
            pair_tensor = imgs_pair.view(b * (t - 1), c * 2, h, w)
            #traj, traj_stdev = self.pose_3d(pair_tensor)  # b*t-1 x 6
            traj_mean, traj_var = self.pose_3d(pair_tensor)
            traj_stdev = torch.exp(0.5 * traj_var)
            eps = torch.randn_like(traj_stdev)
            if self.use_vae:
                traj = traj_mean + eps * traj_stdev
            else:
                traj = traj_mean
            poses = torch.cat([torch.zeros(b, 1, 6).cuda(), traj.view(b, t - 1, 6)], dim=1).view(b * t, 6)

            poses_for_interp = poses.clone().view(b, t, -1).unsqueeze(1).repeat(1, a.size(1), 1, 1)
            a_i = a.view(-1).unsqueeze(1).repeat(1, 6).to(torch.float32)
            poses_for_interp = poses_for_interp.view(-1, t, 6)
            interp_poses = (1 - a_i) * poses_for_interp[:, 0] + a_i * poses_for_interp[:, 1]

            new_poses = interp_poses
            theta = euler2mat(new_poses, scaling=False, translation=True)
            rot_codes = self.rotate_3d(object_code_t, theta)
            output = self.decoder_3d(rot_codes)

            output = F.interpolate(output, (h, w), mode='bilinear')  # T*B x 3 x H x W
            if writer is not None:
                writer.add_text(f"First Last Poses (Testing)", str(poses), step)
                writer.add_text(f"Interpolated Poses (Testing)", str(interp_poses), step)
                writer.add_images(f'Input Images (Testing)', imgs[0], step)
                writer.add_images(f'Interpolated Images (Testing)',
                                torch.clamp(output.view(b, a.size(1), c, h, w)[0], 0, 1), step)
                writer.add_video(f'Interpolation Video (Testing)',
                                torch.clamp(output, 0, 1).view(b, a.size(1), c, h, w)[0].unsqueeze(0), step)

    def update(self, replay_buffer, L,  step=None, writer=None):
        if step % self.update_freq != 0:
            return

        obs, action, reward, next_obs = replay_buffer.sample()
        """
        obs, and next_obs are already normalized
        """
        obs = self.aug(obs)
        next_obs = self.aug(next_obs)
        imgs = obs.clone() # To train 3D

        obs = obs[:, :3]
        next_obs = next_obs[:, :3]

        if self.prop_to_3d:
            obs, _ = self.encoder_3d(obs)
        else:
            with torch.no_grad():
                obs, _ = self.encoder_3d(obs)

        obs = self.encoder_rl(obs)
        with torch.no_grad():
            next_obs, _ = self.encoder_3d(next_obs)
            next_obs = self.encoder_rl(next_obs)

        if self.train_rl:
            self.update_critic(obs, action, reward, next_obs, L, writer, step)
            self.update_actor_and_alpha(obs.detach(), L, writer, step)
            utils.soft_update_params(self.critic, self.critic_target, self.tau)

        freq_3d = self.update_freq * self.update_3d_freq
        if self.train_3d and step % freq_3d == 0:
            n, c, h, w = imgs.shape
            imgs = imgs.view(n, 2, c // 2, h, w)
            imgs = imgs[: self.bsize_3d]
            self.update_3d_recon(imgs, L, writer, step)

