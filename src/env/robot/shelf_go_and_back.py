import numpy as np
import os
import env.robot.reward_utils as reward_utils
from gym import utils
from env.robot.base import BaseEnv, get_full_asset_path


class ShelfGoAndBackEnv(BaseEnv, utils.EzPickle):
    """
    Place the object on the shelf,
    then place the object back to the start
    """

    def __init__(self, xml_path, cameras, n_substeps=20, observation_type='image', reward_type='dense', image_size=84, use_xyz=False, render=False):
        self.sample_large = 1
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
            has_object=True)

        self.state_dim = (26,) if self.use_xyz else (20,)
        self.flipbit, self.task_stage = 1, 1
        utils.EzPickle.__init__(self)

    def compute_reward_stage1(self, achieved_goal, goal, info):
        """
        Reward for the stage one
        (the same with reward in ShelfPlacing)
        """
        actions = self.current_action

        objPos = achieved_goal

        rightFinger, leftFinger = self.sim.data.get_body_xpos(
            'right_hand'), self.sim.data.get_body_xpos('left_hand')
        fingerCOM = (rightFinger + leftFinger)/2

        heightTarget = self.sim.data.get_site_xpos('target0')[2]
        placingGoal = self.sim.data.get_site_xpos('target0')

        reachDist = np.linalg.norm(objPos - fingerCOM)

        placingDist = np.linalg.norm(objPos - placingGoal)

        def reachReward():
            reachRew = -reachDist
            reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
            zRew = np.linalg.norm(fingerCOM[-1] - self.init_finger_xpos[-1])

            if reachDistxy < 0.05:
                reachRew = -reachDist
            else:
                reachRew = -reachDistxy - 2*zRew

            # incentive to close fingers when reachDist is small
            if reachDist < 0.05:
                reachRew = -reachDist + max(actions[-1], 0)/50
            return reachRew, reachDist

        def pickCompletionCriteria():
            tolerance = 0.01
            return objPos[2] >= (heightTarget - tolerance)

        self.pickCompleted = pickCompletionCriteria()

        def objDropped():
            return (objPos[2] < (self.objHeight + 0.005)) and (placingDist > 0.02) and (reachDist > 0.02)
            # Object on the ground, far away from the goal, and from the gripper
            # Can tweak the margin limits

        def orig_pickReward():
            hScale = 100
            if self.pickCompleted and not(objDropped()):
                return hScale*heightTarget
            elif (reachDist < 0.1) and (objPos[2] > (self.objHeight + 0.005)):
                return hScale * min(heightTarget, objPos[2])
            else:
                return 0

        def placeReward():
            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            cond = self.pickCompleted and (reachDist < 0.1) and not(objDropped())

            if cond:
                placeRew = 1000*(self.maxPlacingDist - placingDist) + c1 * \
                                    (np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))
                placeRew = max(placeRew, 0)
                return [placeRew, placingDist]
            else:
                return [0, placingDist]

        reachRew, reachDist = reachReward()
        pickRew = orig_pickReward()
        placeRew, placingDist = placeReward()
        assert ((placeRew >= 0) and (pickRew >= 0))
        reward = reachRew + pickRew + placeRew

        # enter next task stage
        # FIXME: the threshold can be adjusted.
        if placingDist < 0.01:
            self.task_stage += 1
            self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) \
                            - np.array(self._target_pos)) + self.heightTarget

        return reward


    def compute_reward_stage2(self, achieved_goal, goal, info):
        """
        Reward for the stage two
        """
        actions = self.current_action

        objPos = achieved_goal

        rightFinger, leftFinger = self.sim.data.get_body_xpos('right_hand'), self.sim.data.get_body_xpos('left_hand')
        fingerCOM  =  (rightFinger + leftFinger)/2

        goal = self._target_pos
        heightTarget = goal[2]
        placingGoal = goal

        reachDist = np.linalg.norm(objPos - fingerCOM)

        placingDist = np.linalg.norm(objPos - placingGoal)


        def reachReward():
            """
            Reach the object
            """
            reachRew = -reachDist
            reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
            zRew = np.linalg.norm(fingerCOM[-1] - self.init_finger_xpos[-1])

            if reachDistxy < 0.05:
                reachRew = -reachDist
            else:
                reachRew =  -reachDistxy - 2*zRew

            # incentive to close fingers when reachDist is small
            if reachDist < 0.05:
                reachRew = -reachDist + max(actions[-1],0)/50
            return reachRew , reachDist

        def graspCompletionCriteria():
            tolerance = 0.05
            return reachDist < tolerance

        self.graspCompleted = graspCompletionCriteria()


        def objDropped():
            return (objPos[2] < (self.objHeight + 0.005)) and (placingDist >0.02) and (reachDist > 0.02)
            # Object on the ground, far away from the goal, and from the gripper
            # Can tweak the margin limits

        def orig_pickReward():
            hScale = 100
            if self.graspCompleted and not(objDropped()):
                # return hScale*heightTarget
                return hScale*3
            elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)):
                return hScale* min(heightTarget, objPos[2])
            else:
                return 0

        def placeReward():
            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            cond = self.graspCompleted and (reachDist < 0.1) and not(objDropped())

            if cond:
                placeRew = 1000*(self.maxPlacingDist - placingDist) + c1*(np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))
                placeRew = max(placeRew,0)
                return [placeRew , placingDist]
            else:
                return [0 , placingDist]

        def stageReward():
            """
            Reward that the agent enters the second stage 
            """
            if not(objDropped()):
                return 1000
            else:
                return -100

        reachRew, reachDist = reachReward()
        pickRew = orig_pickReward()
        placeRew , placingDist = placeReward()
        assert ((placeRew >=0) and (pickRew>=0))
        stageRew = stageReward()
        reward = reachRew + pickRew + placeRew + stageRew


        return reward



    def compute_reward(self, achieved_goal, goal, info):
        if self.task_stage == 1:
            return self.compute_reward_stage1(achieved_goal, goal, info)
        elif self.task_stage == 2:
            return self.compute_reward_stage2(achieved_goal, goal, info)
        else:
            raise Exception('Error: Current stage exceed in shelf go and back.')

    def _get_state_obs(self):
        cot_pos = self.center_of_table.copy()
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep

        eef_pos = self.sim.data.get_site_xpos('grasp')
        eef_velp = self.sim.data.get_site_xvelp('grasp') * dt
        goal_pos = self.goal
        gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint')

        obj_pos = self.sim.data.get_site_xpos('object0')
        obj_rot = self.sim.data.get_joint_qpos('object0:joint')[-4:]
        obj_velp = self.sim.data.get_site_xvelp('object0') * dt
        obj_velr = self.sim.data.get_site_xvelr('object0') * dt

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
        self.over_obj = False
        self.lifted = False # reset stage flag
        self.placed = False # reset stage flag
        self.over_goal = False

        self.task_stage = 1 # reset task stage

        return BaseEnv._reset_sim(self)

    def _set_action(self, action):
        assert action.shape == (4,)

        if self.flipbit:
            action[3] = 0
            self.flipbit = 0
        else:
            action[:3] = np.zeros(3)
            self.flipbit = 1
        
        BaseEnv._set_action(self, action)
        self.current_action = action # store current_action

    def _get_achieved_goal(self):
        """
        Get the position of the target pos.
        """
        return np.squeeze(self.sim.data.get_site_xpos('object0').copy())

    def _sample_object_pos(self):
        """
        Sample the initial position of the object
        """
        
        object_xpos = self.center_of_table.copy() - np.array([0.3, 0, 0])
        object_xpos[0] += self.np_random.uniform(-0.05, 0.05, size=1)
        object_xpos[1] += self.np_random.uniform(-0.1, 0.1, size=1)
        object_xpos[2] += 0.08

        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        object_quat = object_qpos[-4:]

        assert object_qpos.shape == (7,)
        object_qpos[:3] = object_xpos[:3] # 0,1,2 is x,y,z
        object_qpos[-4:] = object_quat # 
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        
        self.obj_init_pos = object_xpos # store this position, used in the reward
        self.objHeight = self.obj_init_pos[2]
        
        
        self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) \
                            - np.array(self._target_pos)) + self.heightTarget


    def _sample_goal(self, new=True):
        """
        Sample the position of the shelf,
        and the goal is set respectively for two stages.
        """
        shelf_qpos = self.sim.data.get_joint_qpos('shelf:joint') 
        shelf_xpos = shelf_qpos[:3]
        shelf_quat = shelf_qpos[-4:]
        
        if new:
            # randomize the position of the shelf
            shelf_xpos[0] += self.np_random.uniform(-0.01  - 0.01 * self.sample_large, 0.01 + 0.01 * self.sample_large, size=1)
            # shelf_xpos[1] += self.np_random.uniform(-0.01 - 0.01 * self.sample_large, 0.01 + 0.01 * self.sample_large, size=1)
            shelf_xpos[1] = self.sim.data.get_site_xpos('object0')[1]

            shelf_qpos[:3] = shelf_xpos
            shelf_qpos[-4:] = shelf_quat

            self.sim.data.set_joint_qpos('shelf:joint', shelf_qpos)
            
            self.task_stage = 1 # init
        else:
            pass
        
        
        
        self.lift_height = 0.15

        if self.task_stage==1:
            goal = self.sim.data.get_site_xpos('target0') # origin: [1.73380063, 0.25477763, 0.78030275]
            self._target_pos = self.sim.data.get_site_xpos('target0')
        elif self.task_stage==2:
            goal = self.obj_init_pos
            self._target_pos = self.obj_init_pos
        else:
            raise Exception('Error: Current stage exceed in shelf go and back.')
        
        self.heightTarget = goal[2]

        return BaseEnv._sample_goal(self, goal)

    def _sample_initial_pos(self):
        """
        Sample the initial position of arm
        """
        gripper_target = np.array([1.2561169, 0.3, 0.69603332])
        gripper_target[0] += self.np_random.uniform(-0.05, 0.1, size=1)
        gripper_target[1] += self.np_random.uniform(-0.1, 0.1, size=1)
        if self.use_xyz:
            gripper_target[2] += self.np_random.uniform(-0.05, 0.05, size=1)
        BaseEnv._sample_initial_pos(self, gripper_target)
