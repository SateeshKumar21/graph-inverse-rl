import gym
import numpy as np
import torch
import graphirl.common as common
from torchkit import checkpoint
import pickle
import os
import collections
from sklearn.metrics import pairwise_distances
import cv2

IMG_HEIGHT, IMG_WIDTH = 448, 448

def load_model(pretrained_path, load_goal_emb):
  """Load a pretrained model and optionally a precomputed goal embedding."""
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  config = common.load_config_from_dir(pretrained_path)
  model = common.get_model(config)
  checkpoint_dir = os.path.join(pretrained_path, "checkpoints")
  checkpoint_manager = checkpoint.CheckpointManager(
      checkpoint.Checkpoint(model=model), checkpoint_dir, device)
  global_step = checkpoint_manager.restore_or_initialize()

#  model.load_state_dict(torch.load(checkpoint_dir + "/0000000000008001.ckpt"))
  if load_goal_emb:
    print("Loading goal embedding.")
    
    with open(os.path.join(pretrained_path, "goal_emb.pkl"), "rb") as fp:
      goal_emb = pickle.load(fp)
    
    with open(os.path.join(pretrained_path, "distance_scale.pkl"), "rb") as fp:
      distance_scale = pickle.load(fp)

    model.goal_emb = goal_emb
    model.distance_scale = distance_scale
  
  return config, model


class DistanceToGoalVisualRewardXARM(gym.Wrapper):
  """Replace the environment reward with distances in embedding space."""

  def __init__(
      self,
      env,
      model,
      goal_emb,
      distance_scale,
      img_size
  ):
    """Constructor.
    Args:
      env: A gym env.
      model: A model that ingests RGB frames and returns embeddings.
      device: Compute device.
      goal_emb: The goal embedding of shape (D,).
      res_hw: Optional (H, W) to resize the environment image before feeding it
        to the model.
      distance_func: Optional function to apply on the embedding distance.
    """
    super().__init__(env)
    print("TCC WRAPPER initialized")
    self._device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self._model = model.to(self._device).eval()
    self._goal_emb = goal_emb
    self._distance_scale = distance_scale
    self.img_size = img_size
    self.centre_of_table = np.array([1.655, 0.3])
    self.gripper_size = 0.025
    self.obj_size = 0.05
    self.goal_size = 0.05
    self.state_dims = 3 
    self.state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dims,), dtype=np.float32)
    self.state_space_shape = self.state_space.shape


  def _to_tensor(self, x):
    #x = torch.from_numpy(x).permute(2, 0, 1).float()[None, None, Ellipsis]
    x = x[:3, :, :]
    x = torch.from_numpy(x).float()[None, None, Ellipsis]
    x = x / 255.0
    x = x.to(self._device)
    return x


  def _get_reward_from_image(self, image):
    """Forward the pixels through the model and compute the reward."""
    image_tensor = self._to_tensor(image)
    emb = self._model.infer(image_tensor).numpy().embs
    dist = np.linalg.norm(emb - self._goal_emb)
   
    dist = -1.0 * dist
    dist *= self._distance_scale
    return dist

  def step(self, action):
    obs, state, env_reward, done, info = self.env.step(action)
    img = np.array(obs)

    learned_reward = self._get_reward_from_image(img)

    obs = cv2.resize(np.transpose(img, (1, 2, 0)), (self.img_size, self.img_size))
    obs = np.transpose(obs, (2, 0, 1))
    info['env_reward'] = env_reward
    state = np.concatenate([state[:3], np.array([state[-1]])]).astype('float64')

    return obs, state, learned_reward, done, info

  def reset(self):
    
    obs, state_obs = self.env.reset()
    obs = cv2.resize(np.transpose(np.array(obs), (1, 2, 0)), (self.img_size, self.img_size))
    obs = np.transpose(obs, (2, 0, 1))
    state_obs = np.concatenate([state_obs[:3], np.array([state_obs[-1]])]).astype('float64')

  
    return obs, state_obs

class DistanceToGoalBboxRewardXARM(DistanceToGoalVisualRewardXARM):

    def __init__(
      self,
      env,
      model,
      goal_emb,
      distance_scale,
      img_size
    ):

      super().__init__(env=env, model=model, goal_emb= goal_emb, distance_scale = distance_scale, img_size = img_size)

      self.centre_of_table = np.array([1.655, 0.3])
      self.gripper_size = 0.025
      self.obj_size = 0.05
      self.goal_size = 0.05


    def _get_reward_from_image(self, image, state=None):
    
      """Forward the pixels through the model and compute the reward."""

      bboxes_curr = self.state_to_bboxes(state)
      bboxes_curr = self.append_distances(bboxes_curr)

      bboxes = torch.from_numpy(bboxes_curr).float()[None, None, Ellipsis]
      
      bboxes = bboxes.to(self._device)
      emb = self._model.infer(bboxes).numpy().embs
      dist = np.linalg.norm(emb - self._goal_emb)
     
      dist = -1.0 * dist
      dist *= self._distance_scale
      return dist

    def append_distances(self, bboxes):
      
      bboxes = bboxes[0]
      centres_y, centres_x = (bboxes[:, 3] + bboxes[:, 1])/2, (bboxes[:, 2] + bboxes[:, 0])/2 
      centres = np.column_stack([centres_x, centres_y])
      distances = pairwise_distances(centres)

      final_features = np.column_stack([bboxes, distances])

      return np.reshape(final_features, (1, 3, 7))
  

    def vis_bboxes(self, bboxes):
      img = np.ones((224, 224, 3)) * 255
      for i in range(bboxes.shape[0]):
        x1, y1, x2, y2 = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]

        img = cv2.rectangle(img, (int(x1*224), int(y1*224)), (int(x2*224), int(y2*224)), (255, 0, 0), 2)
      
      return img

    def step(self, action):
    
      obs, state, env_reward, done, info = self.env.step(action)

      learned_reward = self._get_reward_from_image(image=None, state=state) #+ env_reward

      obs = cv2.resize(np.transpose(np.array(obs), (1, 2, 0)), (self.img_size, self.img_size))
      obs = np.transpose(obs, (2, 0, 1))
      info['env_reward'] = env_reward
      state = np.concatenate([state[:3], np.array([state[-1]])]).astype('float64')

      return obs, state, learned_reward, done, info

    def state_to_bboxes(self, state):
       
      bboxes = np.zeros((3, 4))
      gripper_pos = self.centre_of_table - state[:2] 
      obj_pos = self.centre_of_table - state[6:8]
      goal_pos = self.centre_of_table - state[4:6]
      gripper_pos = self.convert_cordinates(gripper_pos)
      obj_pos = self.convert_cordinates(obj_pos)
      goal_pos = self.convert_cordinates(goal_pos)
      

      bboxes[0] = gripper_pos[0] - self.gripper_size, gripper_pos[1] - self.gripper_size, gripper_pos[0] + self.gripper_size, gripper_pos[1] + self.gripper_size
      bboxes[1] = obj_pos[0] - 0.05, obj_pos[1] - self.obj_size, obj_pos[0] + self.obj_size, obj_pos[1] + self.obj_size
      bboxes[2] = goal_pos[0] - self.goal_size, goal_pos[1] - self.goal_size, goal_pos[0] + self.goal_size, goal_pos[1] + self.goal_size
      bboxes =  np.reshape(bboxes, (1, 3, 4)) 
      return bboxes

    def convert_cordinates(self, pos):

      pos = pos + 0.5

      return pos

    def reset(self):
    
      obs, state_obs = self.env.reset()
      obs = cv2.resize(np.transpose(np.array(obs), (1, 2, 0)), (self.img_size, self.img_size))
      obs = np.transpose(obs, (2, 0, 1))
      state_obs = np.concatenate([state_obs[:3], np.array([state_obs[-1]])]).astype('float64') #3d state

    
      return obs, state_obs

class DistanceToGoalBboxRewardXARMPegbox(DistanceToGoalVisualRewardXARM):

    def __init__(
      self,
      env,
      model,
      goal_emb,
      distance_scale,
      img_size
    ):

      super().__init__(env=env, model=model, goal_emb= goal_emb, distance_scale = distance_scale, img_size = img_size)
      self.centre_of_table = np.array([1.655, 0.3])
      self.gripper_size = 0.02
      self.obj_size = 0.03
      self.goal_size = 0.03 #0.08

    
    def project_points_from_world_to_camera(self, points, camera_height, camera_width):
      """
      Helper function to project a batch of points in the world frame
      into camera pixels using the world to camera transformation.
      Args:
          points (np.array): 3D points in world frame to project onto camera pixel locations. Should
              be shape [..., 3].
          world_to_camera_transform (np.array): 4x4 Tensor to go from robot coordinates to pixel
              coordinates.
          camera_height (int): height of the camera image
          camera_width (int): width of the camera image
      Return:
          pixels (np.array): projected pixel indices of shape [..., 2]
      """
      world_to_camera_transform = np.load("src/Transformation_matrix_448.npy")
      

      assert points.shape[-1] == 3  # last dimension must be 3D
      assert len(world_to_camera_transform.shape) == 2
      assert world_to_camera_transform.shape[0] == 4 and world_to_camera_transform.shape[1] == 4

      # convert points to homogenous coordinates -> (px, py, pz, 1)
      ones_pad = np.ones(points.shape[:-1] + (1,))
      points = np.concatenate((points, ones_pad), axis=-1)  # shape [..., 4]

      # batch matrix multiplication of 4 x 4 matrix and 4 x 1 vectors to do robot frame to pixels transform
      mat_reshape = [1] * len(points.shape[:-1]) + [4, 4]
      cam_trans = world_to_camera_transform.reshape(mat_reshape)  # shape [..., 4, 4]
      pixels = np.matmul(cam_trans, points[..., None])[..., 0]  # shape [..., 4]

      # re-scaling from homogenous coordinates to recover pixel values
      # (x, y, z) -> (x / z, y / z)
      pixels = pixels / pixels[..., 2:3]
      pixels = pixels[..., :2].round().astype(int)  # shape [..., 2]

      # swap first and second coordinates to get pixel indices that correspond to (height, width)
      # and also clip pixels that are out of range of the camera image
      # pixels = np.concatenate(
      #     (
      #         pixels[..., 1:2].clip(0, camera_height - 1),
      #         pixels[..., 0:1].clip(0, camera_width - 1),
      #     ),
      #     axis=-1,
      # ) # DO NOT remove out of range co-ords

      return pixels

    def state_to_bboxes(self, state):

      # print(f"Shape of state : {state.shape}")
      # print(state[9:12])
      # print(state[6:9])
      gripper_height = 20
      object_height = 25
      goal_height = 45
      #bboxes = np.zeros((3, 4))
      bboxes = np.zeros((2, 4))
      gripper_pos = self.project_points_from_world_to_camera(state[:3], IMG_HEIGHT, IMG_WIDTH)
      object_pos = self.project_points_from_world_to_camera(state[9:12], IMG_HEIGHT, IMG_WIDTH)
      goal_pos = self.project_points_from_world_to_camera(state[6:9], IMG_HEIGHT, IMG_WIDTH) - [10, 0]

      # bboxes[0] = self.construct_bbox(gripper_pos, gripper_height)
      # bboxes[1] = self.construct_bbox(object_pos, object_height)
      # bboxes[2] = self.construct_bbox(goal_pos, goal_height)

      bboxes[0] = self.construct_bbox(object_pos, object_height)
      bboxes[1] = self.construct_bbox(goal_pos, goal_height)

      bboxes = self.normalize_bboxes(bboxes)
    
      return bboxes

    def construct_bbox(self, pos, height):

      return [pos[1] - height, pos[0] - height, pos[1] + height, pos[0] + height]
  

    def normalize_bboxes(self, bboxes):

      bboxes_normalized = np.zeros_like(bboxes)
      for i, bbox in enumerate(bboxes):
          bboxes_normalized[i] = bbox / IMG_WIDTH # assuming image is a square

      return bboxes_normalized

    def _get_reward_from_image(self, image, state=None):
    
      """Forward the pixels through the model and compute the reward."""

      bboxes_curr = self.state_to_bboxes(state)
      bboxes_curr = self.append_distances(bboxes_curr)
     
      bboxes = torch.from_numpy(bboxes_curr).float()[None, None, Ellipsis]
      
      bboxes = bboxes.to(self._device)
      emb = self._model.infer(bboxes).numpy().embs
      dist = np.linalg.norm(emb - self._goal_emb)
     
      dist = -1.0 * dist
      dist *= self._distance_scale
      return dist

    def append_distances(self, bboxes):

      centres_y, centres_x = (bboxes[:, 3] + bboxes[:, 1])/2, (bboxes[:, 2] + bboxes[:, 0])/2 
      centres = np.column_stack([centres_x, centres_y])
      distances = pairwise_distances(centres)

      final_features = np.column_stack([bboxes, distances])

      return np.reshape(final_features, (1, 2, 6))
  

    def vis_bboxes(self, bboxes):
      img = np.ones((224, 224, 3)) * 255
      for i in range(bboxes.shape[0]):
        x1, y1, x2, y2 = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]

        img = cv2.rectangle(img, (int(x1*224), int(y1*224)), (int(x2*224), int(y2*224)), (255, 0, 0), 2)
      
      return img

    def step(self, action):
    
      obs, state, env_reward, done, info = self.env.step(action)

      learned_reward = self._get_reward_from_image(image=None, state=state) #+ env_reward

      obs = cv2.resize(np.transpose(np.array(obs), (1, 2, 0)), (self.img_size, self.img_size))
      obs = np.transpose(obs, (2, 0, 1))
      info['env_reward'] = env_reward
      info['state_full'] = state
      state = np.concatenate([state[:3], np.array([state[-1]])]).astype('float64') #3d state
      
      return obs, state, learned_reward, done, info

    def reset(self):
    
      obs, state_obs = self.env.reset()
      obs = cv2.resize(np.transpose(np.array(obs), (1, 2, 0)), (self.img_size, self.img_size))
      obs = np.transpose(obs, (2, 0, 1))
      state_obs = np.concatenate([state_obs[:3], np.array([state_obs[-1]])]).astype('float64') #3d state

    
      return obs, state_obs


class DistanceToGoalBboxRewardXARMReach(DistanceToGoalVisualRewardXARM):

    def __init__(
      self,
      env,
      model,
      goal_emb,
      distance_scale,
      img_size
    ):

      super().__init__(env=env, model=model, goal_emb= goal_emb, distance_scale = distance_scale, img_size = img_size)

  


    def _get_reward_from_image(self, image, state=None):
    
      """Forward the pixels through the model and compute the reward."""

      bboxes_curr = self.state_to_bboxes(state)
      bboxes_curr = self.append_distances(bboxes_curr)

      bboxes = torch.from_numpy(bboxes_curr).float()[None, None, Ellipsis]
      
      bboxes = bboxes.to(self._device)
      emb = self._model.infer(bboxes).numpy().embs
      dist = np.linalg.norm(emb - self._goal_emb)
     
      dist = -1.0 * dist
      dist *= self._distance_scale
      return dist

    def append_distances(self, bboxes):
      
      bboxes = bboxes[0]
      centres_y, centres_x = (bboxes[:, 3] + bboxes[:, 1])/2, (bboxes[:, 2] + bboxes[:, 0])/2 
      centres = np.column_stack([centres_x, centres_y])
      distances = pairwise_distances(centres)

      final_features = np.column_stack([bboxes, distances])

      return np.reshape(final_features, (1, 2, 6))
  

    def vis_bboxes(self, bboxes):
      img = np.ones((224, 224, 3)) * 255
      for i in range(bboxes.shape[0]):
        x1, y1, x2, y2 = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]

        img = cv2.rectangle(img, (int(x1*224), int(y1*224)), (int(x2*224), int(y2*224)), (255, 0, 0), 2)
      
      return img

    def step(self, action):
    
      obs, state, env_reward, done, info = self.env.step(action)

      learned_reward = self._get_reward_from_image(image=None, state=state) #+ env_reward

      obs = cv2.resize(np.transpose(np.array(obs), (1, 2, 0)), (self.img_size, self.img_size))
      obs = np.transpose(obs, (2, 0, 1))
      info['env_reward'] = env_reward
      state = np.concatenate([state[:2], np.array([state[-1]])]).astype('float64')

      return obs, state, learned_reward, done, info

    def state_to_bboxes(self, state):
       
      bboxes = np.zeros((2, 4))
      gripper_pos = self.centre_of_table - state[:2] 
      goal_pos = self.centre_of_table - state[4:6]
      gripper_pos = self.convert_cordinates(gripper_pos)
      goal_pos = self.convert_cordinates(goal_pos)
      

      bboxes[0] = gripper_pos[0] - self.gripper_size, gripper_pos[1] - self.gripper_size, gripper_pos[0] + self.gripper_size, gripper_pos[1] + self.gripper_size
      bboxes[1] = goal_pos[0] - self.goal_size, goal_pos[1] - self.goal_size, goal_pos[0] + self.goal_size, goal_pos[1] + self.goal_size
      bboxes =  np.reshape(bboxes, (1, 2, 4)) 
      return bboxes

    def convert_cordinates(self, pos):

      pos = pos + 0.5

      return pos

def get_wrapper(reward_wrapper_type, env, img_size, pretrained_path=None):
  """Wrap the environment based on values in the config."""
 
  print(pretrained_path)
  if reward_wrapper_type != "none":
    
    model_config, model = load_model(
        pretrained_path,
        # The goal classifier does not use a goal embedding.
        reward_wrapper_type != "goal_classifier",
    )
    kwargs = {
        "env": env,
        "model": model,
        "goal_emb": model.goal_emb,
        "distance_scale": model.distance_scale,
        "img_size": img_size
    }

    
    if reward_wrapper_type == "tcc":
      env = DistanceToGoalVisualRewardXARM(**kwargs)
    elif reward_wrapper_type == "gil":
      env = DistanceToGoalBboxRewardXARM(**kwargs)
    elif reward_wrapper_type == "gil_reach":
      env = DistanceToGoalBboxRewardXARMReach(**kwargs)
    
    elif reward_wrapper_type == "gil_pegbox":
      env = DistanceToGoalBboxRewardXARMPegbox(**kwargs)
    else:
      raise ValueError(
          f"{reward_wrapper_type} is not a supported reward wrapper.")
  return env

if __name__ == '__main__':
    load_model("/data/sateesh/trained_models/xirl/exp1_trainon=human_maxdemosperemb=1000_uid=630173526", True)