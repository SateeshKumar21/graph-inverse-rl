from gym.envs.registration import register

REGISTERED_ROBOT_ENVS = False


def register_robot_envs(n_substeps=20, observation_type='image', reward_type='dense', image_size=84, use_xyz=False):
	global REGISTERED_ROBOT_ENVS
	if REGISTERED_ROBOT_ENVS:	
		return

	register(
		id='RobotLift-v0',
		entry_point='env.robot.lift:LiftEnv',
		kwargs=dict(
			xml_path='robot/lift.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	register(
		id='RobotPickplace-v0',
		entry_point='env.robot.pick_place:PickPlaceEnv',
		kwargs=dict(
			xml_path='robot/pick_place.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	register(
		id='RobotPegbox-v0',
		entry_point='env.robot.peg_in_box:PegBoxEnv',
		kwargs=dict(
			xml_path='robot/peg_in_box.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
			
		)
	)

	register(
		id='RobotHammer-v0',
		entry_point='env.robot.hammer:HammerEnv',
		kwargs=dict(
			xml_path='robot/hammer.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	register(
		id='RobotHammerall-v0',
		entry_point='env.robot.hammer_all:HammerAllEnv',
		kwargs=dict(
			xml_path='robot/hammer_all.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	register(
		id='RobotReach-v0',
		entry_point='env.robot.reach:ReachEnv',
		kwargs=dict(
			xml_path='robot/reach.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	register(
		id='RobotReachmovingtarget-v0',
		entry_point='env.robot.reach:ReachMovingTargetEnv',
		kwargs=dict(
			xml_path='robot/reach.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	register(
		id='RobotPush-v0',
		entry_point='env.robot.push:PushEnv',
		kwargs=dict(
			xml_path='robot/push.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	register(
		id='RobotPushnogoal-v0',
		entry_point='env.robot.push:PushNoGoalEnv',
		kwargs=dict(
			xml_path='robot/push.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	# --- Shelf Placing Task Class --- #

	# classic view
	register(
		id='RobotShelfplacing-v0',
		entry_point='env.robot.shelf_placing:ShelfPlacingEnv',
		kwargs=dict(
			xml_path='robot/shelf_placing_classic.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	# a near view
	register(
		id='RobotShelfplacingnear-v0',
		entry_point='env.robot.shelf_placing:ShelfPlacingEnv',
		kwargs=dict(
			xml_path='robot/shelf_placing_near.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	# a far view
	register(
		id='RobotShelfplacingfar-v0',
		entry_point='env.robot.shelf_placing:ShelfPlacingEnv',
		kwargs=dict(
			xml_path='robot/shelf_placing_far.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	# a task based on ShelfPlacing
	register(
		id='RobotShelfgoandback-v0',
		entry_point='env.robot.shelf_go_and_back:ShelfGoAndBackEnv',
		kwargs=dict(
			xml_path='robot/shelf_placing_classic.xml', # the same 
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	REGISTERED_ROBOT_ENVS = True
