import argparse
import numpy as np


def parse_args():
	parser = argparse.ArgumentParser()

	# environment
	parser.add_argument('--domain_name', default='robot')
	parser.add_argument('--task_name', default='reach')
	parser.add_argument('--frame_stack', default=1, type=int)
	parser.add_argument('--action_repeat', default=1, type=int)
	parser.add_argument('--episode_length', default=50, type=int)
	parser.add_argument('--n_substeps', default=20, type=int)
	parser.add_argument('--eval_mode', default='none', type=str)
	parser.add_argument('--from_state', default=False, action='store_true')
	parser.add_argument('--action_space', default='xy', type=str)
	parser.add_argument('--cameras', default= 0, type=int) # 0: 3rd person, 1: 1st person, 2: both, 5: dynamic
	parser.add_argument('--render', default=False, type=bool)
	parser.add_argument('--camera_dropout', default=0, type=int) # [0,1,2,3] 0: None, 1: TP, 2: FP, 3: Random
	parser.add_argument('--observation_type', default='image', type=str) # 'state', 'image', 'state+image'
	parser.add_argument('--render_image_size', default=84, type=int)

	# agent
	parser.add_argument('--algorithm', default='sac', type=str)
	parser.add_argument('--train_steps', default='500k', type=str)
	parser.add_argument('--buffer_size', default='500k', type=str)
	parser.add_argument('--discount', default=0.99, type=float)
	parser.add_argument('--init_steps', default=1000, type=int)
	parser.add_argument('--batch_size', default=128, type=int)
	parser.add_argument('--hidden_dim', default=1024, type=int)
	parser.add_argument('--hidden_dim_state', default=128, type=int)
	parser.add_argument('--image_size', default=84, type=int)
	parser.add_argument('--attention', default=1, type=int)
	parser.add_argument('--concat', default=1, type=int)
	parser.add_argument('--context1', default=1, type=int)
	parser.add_argument('--context2', default=1, type=int)
	parser.add_argument('--load_steps', default='500k', type=str)
	parser.add_argument('--predict_state', default=0, type=int)

	# actor
	parser.add_argument('--actor_lr', default=1e-3, type=float)
	parser.add_argument('--actor_beta', default=0.9, type=float)
	parser.add_argument('--actor_log_std_min', default=-10, type=float)
	parser.add_argument('--actor_log_std_max', default=2, type=float)
	parser.add_argument('--actor_update_freq', default=2, type=int)

	# critic
	parser.add_argument('--critic_lr', default=1e-3, type=float)
	parser.add_argument('--critic_beta', default=0.9, type=float)
	parser.add_argument('--critic_tau', default=0.01, type=float)
	parser.add_argument('--critic_target_update_freq', default=2, type=int)

	# architecture
	parser.add_argument('--num_shared_layers', default=11, type=int)
	parser.add_argument('--num_head_layers', default=0, type=int)
	parser.add_argument('--num_filters', default=32, type=int)
	parser.add_argument('--projection_dim', default=100, type=int)
	parser.add_argument('--encoder_tau', default=0.05, type=float)
	
	# entropy maximization
	parser.add_argument('--init_temperature', default=0.1, type=float)
	parser.add_argument('--alpha_lr', default=1e-4, type=float)
	parser.add_argument('--alpha_beta', default=0.5, type=float)

	# auxiliary tasks
	parser.add_argument('--aux_lr', default=1e-3, type=float)
	parser.add_argument('--aux_beta', default=0.9, type=float)
	parser.add_argument('--aux_update_freq', default=2, type=int)

	#graphirl

	parser.add_argument('--pretrained_path', type=str)
	parser.add_argument('--reward_wrapper', type=str, default="gil") # tcc -> XIRL, gil -> GIL
	parser.add_argument('--debug', default=False, action="store_true", help="for logging to a debug folder")
	parser.add_argument('--apply_wrapper', default=False, action="store_true")
	

	# soda
	parser.add_argument('--soda_batch_size', default=256, type=int)
	parser.add_argument('--soda_tau', default=0.005, type=float)

	# svea
	parser.add_argument('--use_vit', default=False, action='store_true')
	parser.add_argument('--svea_num_heads', default=8, type=int)
	parser.add_argument('--svea_embed_dim', default=128, type=int)
	parser.add_argument('--svea_alpha', default=0.5, type=float)
	parser.add_argument('--svea_beta', default=0.5, type=float)
	parser.add_argument('--svea_augmentation', default='colorjitter', type=str) # 'colorjitter' or 'affine+colorjitter' or 'noise' or 'affine+noise' or 'conv' or 'affine+conv'
	parser.add_argument('--naive', default=False, action='store_true') # apply data aug naively

	# sacv2 / ddpg (drqv2)
	parser.add_argument('--lr', default=1e-3, type=float) # single learning rate for all modules (sacv2 / ddpg)
	parser.add_argument('--update_freq', default=2, type=int) # single update frequency for all losses (sacv2 / ddpg)
	parser.add_argument('--tau', default=0.01, type=float) # single soft target update rate for encoder and critic (sacv2 / ddpg)
	parser.add_argument('--num_expl_steps', default=1000, type=int) # number of uniformly distributed steps for exploration (ddpg)
	parser.add_argument('--std_schedule', default='linear(1.0,0.1,0.25)', type=str) # linear(initial, final, % of train steps) or float
	parser.add_argument('--std_clip', default=0.3, type=float) # std clipping (ddpg)
	parser.add_argument('--mean_zero', default=False, action='store_true') # normalize images to range [-0.5, 0.5] instead of [0, 1] (all)

	# sacv2_3d
	parser.add_argument('--train_rl', default=False, action='store_true')
	parser.add_argument('--train_3d', default=False, action='store_true')
	parser.add_argument('--prop_to_3d', default=False, action='store_true')
	parser.add_argument('--bottleneck', default=16, type=int)
	parser.add_argument('--lr_3dc', default=1e-3, type=float)
	parser.add_argument('--lr_3dp', default=1e-3, type=float)
	parser.add_argument('--log_3d_imgs', default="15k", type=str)
	parser.add_argument('--log_train_video', default="50k", type=str)
	parser.add_argument('--only_2_recon', default=False, action='store_true')
	parser.add_argument('--rand_first', default=False, action='store_true')
	parser.add_argument('--use_vae', default=False, action='store_true')
	parser.add_argument('--huber', default=False, action='store_true')
	parser.add_argument('--bsize_3d', default=8, type=int)
	parser.add_argument('--update_3d_freq', default=1, type=int)
	parser.add_argument('--use_latent', default=False, action='store_true')
	parser.add_argument('--use_impala', default=False, action='store_true')
	parser.add_argument('--project_conv', default=False, action='store_true')

	# eval
	parser.add_argument('--save_freq', default='100k', type=str)
	parser.add_argument('--eval_freq', default='10k', type=str)
	parser.add_argument('--eval_episodes', default=1, type=int)

	# misc
	parser.add_argument('--seed', default=None, type=int)
	parser.add_argument('--exp_suffix', default='default', type=str)
	parser.add_argument('--log_dir', default='logs', type=str)
	parser.add_argument('--save_video', default=False, action='store_true')
	parser.add_argument('--num_seeds', default=1, type=int)
	parser.add_argument('--visualize_configurations', default=False, action='store_true')
	parser.add_argument('--visualize_augmentations', default=False, action='store_true')
	
	# wandb's setting
	parser.add_argument('--wandb', default=False, action='store_true')
	parser.add_argument('--wandb_project', default='robot_project', type=str)
	parser.add_argument('--wandb_name', default=None, type=str)
	parser.add_argument('--wandb_group', default=None, type=str)
	parser.add_argument('--wandb_job', default=None, type=str)


	args = parser.parse_args()

	assert args.algorithm in {'sac', 'sacv2', 'sacv2_3d', 'drq', 'sveav2', 'drqv2', 'multiview', 'drq_multiview'}, f'specified algorithm "{args.algorithm}" is not supported'
	assert args.image_size in {64, 84, 128}, f'image size = {args.image_size} is strongly discouraged, use one of [64, 84, 128] (default: 84)'
	assert not args.use_vit, f'use_vit should not be used'
	assert args.svea_augmentation in {'colorjitter', 'affine+colorjitter', 'noise', 'affine+noise', 'conv', 'affine+conv', 'overlay', 'affine+overlay', 'none'}, f'svea_augmentation = {args.svea_augmentation} is not supported'
	assert args.action_space in {'xy', 'xyz', 'xyzw'}, f'specified action_space "{args.action_space}" is not supported'
	assert args.eval_mode in {'train', 'test' ,'color_easy', 'color_hard', 'video_easy', 'video_hard', 'none', None}, f'specified mode "{args.eval_mode}" is not supported'
	assert args.seed is not None, 'must provide seed for experiment'
	assert args.exp_suffix is not None, 'must provide an experiment suffix for experiment'
	assert args.log_dir is not None, 'must provide a log directory for experiment'
	assert (args.algorithm=='sacv2_3d' and args.cameras==5) or (args.algorithm!='sacv2_3d' and args.cameras!=5), 'dynamic pose and sacv2_3d should appear at the same time'

	args.train_steps = int(args.train_steps.replace('k', '000'))
	args.buffer_size = int(args.buffer_size.replace('k', '000'))
	args.save_freq = int(args.save_freq.replace('k', '000'))
	args.eval_freq = int(args.eval_freq.replace('k', '000'))
	args.log_3d_imgs = int(args.log_3d_imgs.replace('k', '000'))
	args.log_train_video = int(args.log_train_video.replace('k', '000'))
	
	if args.eval_mode == 'none':
		args.eval_mode = None
	
	return args
