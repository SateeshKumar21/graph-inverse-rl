from algorithms.sac import SAC
from algorithms.sacv2 import SACv2
from algorithms.drq import DrQ
from algorithms.sveav2 import SVEAv2
from algorithms.multiview import MultiView
from algorithms.drq_multiview import DrQMultiView
from algorithms.drqv2 import DrQv2
from algorithms.sacv2_3d import SACv2_3D

algorithm = {
	'sac': SAC,
	'sacv2': SACv2,
	'sacv2_3d':SACv2_3D,
	'drq': DrQ,
	'sveav2': SVEAv2,
	'multiview': MultiView,
	'drq_multiview': DrQMultiView,
	'drqv2': DrQv2
}


def make_agent(obs_shape, state_shape , action_shape, args):
	if args.algorithm=='sacv2_3d':
		if args.use_latent:
			a_obs_shape = (args.bottleneck*32, 32, 32)
		elif args.use_impala:
			a_obs_shape = (32, 8, 8)
		else:
			a_obs_shape = (32, 26, 26)
		return algorithm[args.algorithm](a_obs_shape, (3, 64, 64), action_shape, args)
	else:
		return algorithm[args.algorithm](obs_shape, state_shape, action_shape, args)