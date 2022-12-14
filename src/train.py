import torch
import os

import numpy as np
import gym
import utils
import time
import wandb
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder
from state import StateRecorder
from graphirl_wrapper import get_wrapper


torch.backends.cudnn.benchmark = True

def evaluate(env, agent, video, state_recorder, num_episodes, L, step, test_env=False, use_wandb=False):
	episode_rewards = []
	environment_rewards = []
	success_rate = []
	success_rate_05 = []
	success_rate_15 = []
	final_distance = []
	for i in range(num_episodes):
		obs, state = env.reset()
		video.init(enabled=(i==0))
		state_recorder.init(enabled=(i==0))
		
		done = False
		episode_reward = 0
		environment_reward = 0
		while not done:
			with torch.no_grad(), utils.eval_mode(agent):
				
				action = agent.select_action(obs, state)
			obs, state, reward, done, info = env.step(action)
			video.record(env)
			if 'state_full' in info.keys():
				#print("Logging state full")
				state_recorder.record(info['state_full'])
			episode_reward += reward
			if 'env_reward' in info.keys():
				environment_reward += info['env_reward']
			else:
				environment_reward += reward
		if 'distance' in info.keys():
			final_distance.append(info['distance'])
		
		if L is not None:
			_test_env = '_test_env' if test_env else ''
			video.save(f'{step}{_test_env}.mp4')
			state_recorder.save(f'{step}{_test_env}.npy')
			
		if args.wandb and i==0:
			# utils.save_image(torch.tensor(video.frames[0].transpose(2, 0, 1)), 'test.png')
			frames = np.array([frame.transpose(2, 0, 1)  for frame in video.frames])
			wandb.log({'eval/eval_video%s'%_test_env: wandb.Video(frames, fps=video.fps, format="mp4") }, step=step+1)

		
		episode_rewards.append(episode_reward)
		environment_rewards.append(environment_reward)
		if 'is_success' in info:
			#print("HERE in is_success")
			success = float(info['is_success'])
			success_rate.append(success)

		if 'success_rate_05' in info:
			#print("HERE in is_success_05")
			success_2 = float(info['success_rate_05'])
			success_rate_05.append(success_2)
		

		if 'success_rate_15' in info:
			#print("HERE in is_success_05")
			success_3 = float(info['success_rate_15'])
			success_rate_15.append(success_3)
	
	
	L.log(f'eval/sucess_rate', np.nanmean(success_rate), step)
	if args.wandb:
		wandb.log({'eval/sucess_rate_10':np.nanmean(success_rate)}, step=step+1)
		wandb.log({'eval/sucess_rate_05':np.nanmean(success_rate_05)}, step=step+1)
		wandb.log({'eval/sucess_rate_15':np.nanmean(success_rate_15)}, step=step+1)

	L.log(f'eval/episode_reward{_test_env}', np.mean(episode_rewards), step)
	L.log(f'eval/environment_reward{_test_env}', np.mean(environment_rewards), step)
	if args.wandb:
		wandb.log({'eval/episode_reward':np.mean(episode_rewards)}, step=step+1)
		wandb.log({'eval/environment_reward':np.mean(environment_rewards)}, step=step+1)
		wandb.log({'eval/final_environment_distance': np.mean(final_distance)}, step=step+1)


	return np.nanmean(episode_rewards), np.nanmean(success_rate)



def visualize_configurations(env, args):
	from torchvision.utils import make_grid, save_image
	frames = []
	for i in range(20):
		env.reset()
		frame = torch.from_numpy(env.render_obs(mode='rgb_array', height=448, width=448, camera_id=0).copy()).squeeze(0)
		frame = frame.permute(2,0,1).float().div(255)
		frames.append(frame)
	
	save_image(make_grid(torch.stack(frames), nrow=5), f'{args.domain_name}_{args.task_name}.png')

def visualize_augmentations(env, args):
	from torchvision.utils import make_grid, save_image
	import augmentations
	frames = []
	for aug in [augmentations.random_color_jitter,
				lambda x: augmentations.random_color_jitter(augmentations.random_affine(x)),
				augmentations.random_noise,
				lambda x: augmentations.random_noise(augmentations.random_affine(x)),
				augmentations.random_conv,
				lambda x: augmentations.random_conv(augmentations.random_affine(x))]:
		for i in range(5):
			obs, state = env.reset()
			frame = torch.FloatTensor(np.array(obs))[-3:].unsqueeze(0).cuda()
			frames.append(aug(frame).div(255).squeeze(0).cpu())
	save_image(make_grid(torch.stack(frames), nrow=5), f'{args.domain_name}_{args.task_name}.png')


def main(args):

	print("HEREEEE")
	print(f"WANDB: {args.wandb}")
	print(f"Project Name: {args.wandb_project}")
	# init wandb
	if args.wandb:
		wandb.init(project=args.wandb_project, name= str(args.exp_suffix) + "_" + str(args.seed), \
		group=args.wandb_group, job_type=args.wandb_job)
		wandb.config.update(args) # save config
		wandb.run.log_code(".")
	# Set seed
	utils.set_seed_everywhere(args.seed)
	if args.cameras==0:
		cameras=['third_person']
	elif args.cameras==1:
		cameras=['first_person']
	elif args.cameras==2:
		cameras = ['third_person', 'first_person']
	elif args.cameras==3:
		cameras = ['shoulder']
	elif args.cameras==4:
		cameras = ['front_diag']
	elif args.cameras==5: # a mode used in sacv2_3d
		cameras= 'dynamic'
	else:
		raise Exception('Current Camera Pose Not Supported.')

	# Initialize environments
	gym.logger.set_level(40)
	env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed,
		episode_length=args.episode_length,
		n_substeps=args.n_substeps,
		frame_stack=args.frame_stack,
		action_repeat=args.action_repeat,
		image_size=args.render_image_size,
		mode='train',
		cameras=cameras, #['third_person', 'first_person']
		render=args.render, # Only render if observation type is state
		camera_dropout=args.camera_dropout,
		observation_type=args.observation_type,
		action_space=args.action_space,
		rand_first=args.rand_first
	)
	
	test_env = make_env (
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42, # Does 42 means the ultimate answer to the universe?
		episode_length=args.episode_length,
		n_substeps=args.n_substeps,
		frame_stack=args.frame_stack,
		action_repeat=args.action_repeat,
		image_size=args.render_image_size,
		mode=args.eval_mode,
		cameras=cameras, #['third_person', 'first_person']
		render=args.render, # Only render if observation type is state
		camera_dropout=args.camera_dropout,
		observation_type=args.observation_type,
		action_space=args.action_space,
		rand_first=args.rand_first
	)
	
	if args.apply_wrapper:

		env = get_wrapper(args.reward_wrapper, env, args.image_size, pretrained_path = args.pretrained_path)
		test_env = get_wrapper(args.reward_wrapper, test_env, args.image_size, pretrained_path=  args.pretrained_path)
		observation_space = (3, env.img_size, env.img_size)
	#observation_space = (3, 84, 84)
	# Visualize initial configurations
	else:
		assert args.image_size == args.render_image_size, "Resolution must be same if not using gil wrapper"
		observation_space = (3, args.image_size, args.image_size)

	
	# Visualize initial configurations
	if args.visualize_configurations:
		visualize_configurations(env, args)

	# Visualize augmentations
	if args.visualize_augmentations:
		visualize_augmentations(env, args)

	
	arg_to_text = {
	'00000': 'single3',
	'30000': 'singleSide',
	'40000': 'singleDiag',
	'10000': 'single1',
	'20100': 'multi_concat',
	'21100': 'multi_attention_concat',
	'21000': 'multi_attention_add',
	'20000': 'multi_add',
	'21110': 'multi_context1_concat',
	'21010': 'multi_context1_add',
	'21101': 'multi_context2_concat',
	'21001': 'multi_context2_add',
	'21111': 'multi_context12_concat',
	'21011': 'multi_context12_add',
	}


	try:
		dir_name = arg_to_text[str(args.cameras)+str(args.attention)+str(args.concat)+str(args.context1)+str(args.context2)]
		dir_name = dir_name + '_' + args.svea_augmentation + '_' + args.observation_type
	except:
		dir_name = ''
	if args.observation_type=='state':
		dir_name = 'state'

	# Create working directory
	work_dir = os.path.join(args.log_dir, args.domain_name+'_'+args.task_name, args.algorithm, args.exp_suffix, dir_name ,str(args.seed))
	print('Working directory:', work_dir)
	#assert not os.path.exists(os.path.join(work_dir, 'train.log')) or args.exp_suffix == 'dev', 'specified working directory already exists'
	utils.make_dir(work_dir)
	model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
	video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
	state_dir = utils.make_dir(os.path.join(work_dir, 'state'))

	video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448, fps=15 if args.domain_name == 'robot' else 25)
	state_recorder = StateRecorder(state_dir if args.save_video else None) 

	utils.write_info(args, os.path.join(work_dir, 'info.log'))

	# Prepare agent
	assert torch.cuda.is_available(), 'must have cuda enabled'
	replay_buffer = utils.ReplayBuffer(
		obs_shape=observation_space,
		state_shape=env.state_space_shape,
		action_shape=env.action_space.shape,
		capacity=args.buffer_size,
		batch_size=args.batch_size
	)
	print('Observations:', observation_space)
	print('Action space:', f'{args.action_space} ({env.action_space.shape[0]})')
	print('State Space:', env.state_space_shape)
	print(f"Using Buffer size of {args.buffer_size}")
	
	agent = make_agent(
		obs_shape=observation_space,
		state_shape=env.state_space_shape,
		action_shape=env.action_space.shape,
		args=args
	)


	#agent = torch.load(os.path.join(model_dir, str(args.load_steps)+'.pt'))

	start_step, episode, episode_reward, info, done, episode_success = 0, 0, 0, {}, True, 0
	L = Logger(work_dir)
	start_time = time.time()
	if args.render:
		env.render()

	for step in range(start_step, args.train_steps+1):
		if done:
			if step > start_step:
				L.log('train/duration', time.time() - start_time, step)
				start_time = time.time()
				L.dump(step)

			# Evaluate agent periodically
			if step % args.eval_freq == 0:
				print('Evaluating:', work_dir)
				L.log('eval/episode', episode, step)
				evaluate(env, agent, video, state_recorder, args.eval_episodes, L, step, use_wandb=args.wandb)
				if test_env is not None:
					evaluate(test_env, agent, video, state_recorder, args.eval_episodes, L, step, test_env=True, use_wandb=args.wandb)
				L.dump(step)

				# Evaluate 3D
				if args.train_3d:
					obs = env.reset()

					# Execute one timestep to randomize the camera and environemnt.
					a_eval = env.action_space.sample()
					obs, _, _, _ = env.step(a_eval)
					# Select the camera views
					o1 = obs[:3]
					o2 = obs[3:]
					# Concatenate and convert to torch tensor and add unit batch dimensions
					images_rgb = np.concatenate([np.expand_dims(o1, axis=0),
												 np.expand_dims(o2, axis=0)], axis=0)
					images_rgb = torch.from_numpy(images_rgb).float().cuda().unsqueeze(0).div(255)
					agent.gen_interpolate(images_rgb, None, step)


			# Save agent periodically
			if step > start_step and step % args.save_freq == 0:
				torch.save(agent, os.path.join(model_dir, f'{step}.pt'))

			L.log('train/episode_reward', episode_reward, step)
			L.log('train/success_rate', episode_success/args.episode_length, step)
			if args.wandb:
				wandb.log({'train/episode_reward':episode_reward, \
				'train/success_rate':episode_success/args.episode_length}, step=step+1)

			obs, state = env.reset()
			done = False
			episode_reward = 0
			episode_step = 0
			episode += 1
			episode_success = 0

			L.log('train/episode', episode, step)

		# Sample action and update agent
		if step < args.init_steps:
			action = env.action_space.sample()
		else:
			with torch.no_grad(), utils.eval_mode(agent):
				action = agent.sample_action(obs, state, step)
			num_updates = args.init_steps//args.update_freq if step == args.init_steps else 1
			for i in range(num_updates):
				agent.update(replay_buffer=replay_buffer, L=L, step=step)

		# Take step
		next_obs, next_state, reward, done, info = env.step(action)
		replay_buffer.add(obs, state, action, reward, next_obs, next_state)
		episode_reward += reward
		obs = next_obs
		episode_success+=float(info['is_success'])
		episode_step += 1
		if args.render:
			env.render()

	print('Completed training for', work_dir)


if __name__ == '__main__':
	args = parse_args()
	main(args)
