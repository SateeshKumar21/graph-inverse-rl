# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Video dataset abstraction."""

import logging
import os.path as osp
import random
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

from xirl import frame_samplers
from xirl import transforms
from xirl.file_utils import get_subdirs, load_bboxes_pusher_dists_context
from xirl.file_utils import load_image, load_bboxes, load_bboxes_pusher, load_bboxes_pusher, load_bboxes_ordered, load_image_bbox, load_bboxes_pusher_gp, load_bboxes_pusher_dists_gp, load_bboxes_pusher_dists
from xirl.tensorizers import ToTensor_bbox, ToTensor
from xirl.types import SequenceType
from xirl.data_aug import *

DataArrayPacket = Dict[SequenceType, Union[np.ndarray, str, int]]
DataTensorPacket = Dict[SequenceType, Union[torch.Tensor, str]]


class VideoDataset(Dataset):
  """A dataset for working with videos."""

  def __init__(
      self,
      root_dir,
      frame_sampler,
      augmentor = None,
      max_vids_per_class = -1,
      seed = None,
  ):
    """Constructor.

    Args:
      root_dir: The path to the dataset directory.
      frame_sampler: A sampler specifying the frame sampling strategy.
      augmentor: An instance of transforms.VideoAugmentor. If provided, will
        apply data augmentation to the sampled video data.
      max_vids_per_class: The max number of videos to consider per class. The
        remaining videos are ignored, effectively reducing the total dataset
        size.
      seed: The seed for the rng.

    Raises:
      ValueError: If the root directory is empty.
    """
    super().__init__()

    self._root_dir = root_dir
    self._frame_sampler = frame_sampler
    self._seed = seed
    self._max_vids_per_class = max_vids_per_class
    self._augmentor = augmentor
    self._totensor = ToTensor()

    # Get list of available dirs and ensure that it is not empty.
    self._available_dirs = get_subdirs(
        self._root_dir,
        nonempty=True,
        sort=False,
    )
    if len(self._available_dirs) == 0:  # pylint: disable=g-explicit-length-test
      raise ValueError("{} is an empty directory.".format(root_dir))
    self._allowed_dirs = self._available_dirs

    self.seed_rng()
    self._build_dir_tree()

  def seed_rng(self):
    if self._seed:
      random.seed(self._seed)

  def _build_dir_tree(self):
    """Build a dict of indices for iterating over the dataset."""
    self._dir_tree = {}
    for path in self._allowed_dirs:
      vids = get_subdirs(
          path,
          nonempty=False,
          sort=True,
          sortfunc=lambda x: int(osp.splitext(osp.basename(x))[0]),
      )
      if len(vids) > 0:  # pylint: disable=g-explicit-length-test
        self._dir_tree[path] = vids
    self._restrict_dataset_size()

  def _restrict_dataset_size(self):
    """Restrict the max vid per class or max total vids if specified."""
    if self._max_vids_per_class > 0:
      for vid_class, vid_dirs in self._dir_tree.items():
        self._dir_tree[vid_class] = vid_dirs[:self._max_vids_per_class]

  def restrict_subdirs(self, subdirs):
    """Restrict the set of available subdirectories, i.e.

    classes.

    If using a batch sampler in conjunction with a dataloader, ensure this
    method is called before instantiating the sampler.

    Args:
      subdirs: A list of allowed video classes.

    Raises:
      ValueError: If the restriction leads to an empty directory.
    """
    if not isinstance(subdirs, (tuple, list)):
      subdirs = [subdirs]
    if subdirs:
      len_init = len(self._available_dirs)
      self._allowed_dirs = self._available_dirs
      subdirs = [osp.join(self._root_dir, x) for x in subdirs]
      self._allowed_dirs = list(set(self._allowed_dirs) & set(subdirs))
      if len(self._allowed_dirs) == 0:  # pylint: disable=g-explicit-length-test
        raise ValueError(f"Filtering with {subdirs} returns an empty dataset.")
      len_final = len(self._allowed_dirs)
      logging.debug(  # pylint: disable=logging-format-interpolation
          f"Restricted dataset from {len_init} to {len_final} actions.")
      self._build_dir_tree()
    else:
      logging.debug("Passed in an empty list. No action taken.")

  def _get_video_path(self, class_idx, vid_idx):
    """Return video paths given class and video indices.

    Args:
      class_idx: The index of the action class folder in the dataset directory
        tree.
      vid_idx: The index of the video in the action class folder to retrieve.

    Returns:
      A path to a video to sample in the dataset.
    """
    action_class = list(self._dir_tree)[class_idx]
    return self._dir_tree[action_class][vid_idx]

  def _get_data(self, vid_path):
    """Load video data given a video path.

    Feeds the video path to the frame sampler to retrieve video frames and
    metadata.

    Args:
      vid_path: A path to a video in the dataset.

    Returns:
      A dictionary containing key, value pairs where the key is an enum
      member of `SequenceType` and the value is either an int, a string
      or an ndarray respecting the key type.
    """
    sample = self._frame_sampler.sample(vid_path)
    
    # Load each frame along with its context frames into an array of shape
    # (S, X, H, W, C), where S is the number of sampled frames and X is the
    # number of context frames.
    
    # try:
    #   print(sample["frames"][0])
    # except:

    #     print("ERROR in indexing")
    #     print("Vid path: {}".format(vid_path))
    #     exit()
    
    frames = np.stack([load_image(f) for f in sample["frames"]])
    frames = np.take(frames, sample["ctx_idxs"], axis=0)
    #print(frames.shape)

    # Reshape frames into a 4D array of shape (S * X, H, W, C).
    frames = np.reshape(frames, (-1, *frames.shape[2:]))

    frame_idxs = np.asarray(sample["frame_idxs"], dtype=np.int64)

    return {
        SequenceType.FRAMES: frames,
        SequenceType.FRAME_IDXS: frame_idxs,
        SequenceType.VIDEO_NAME: vid_path,
        SequenceType.VIDEO_LEN: sample["vid_len"],
    }

  def __getitem__(self, idxs):
    vid_paths = self._get_video_path(*idxs)
    #print(vid_paths)
    data_np = self._get_data(vid_paths)
    if self._augmentor:
      data_np = self._augmentor(data_np)
    data_tensor = self._totensor(data_np)
    return data_tensor

  def __len__(self):
    return self.total_vids

  @property
  def num_classes(self):
    """The number of subdirs, i.e. allowed video classes."""
    return len(self._allowed_dirs)

  @property
  def total_vids(self):
    """The total number of videos across all allowed video classes."""
    num_vids = 0
    for vids in self._dir_tree.values():
      num_vids += len(vids)
    return num_vids

  @property
  def dir_tree(self):
    """The directory tree."""
    return self._dir_tree

  def collate_fn(
      self,
      batch,
  ):
    """A custom collate function for video data."""

    def _stack(key):
      return torch.stack([b[key] for b in batch])

    # Convert the keys to their string representation so that a batch can be
    # more easily indexed into without an extra import to SequenceType.
    return {
        str(SequenceType.FRAMES):
            _stack(SequenceType.FRAMES),
        str(SequenceType.FRAME_IDXS):
            _stack(SequenceType.FRAME_IDXS),
        str(SequenceType.VIDEO_LEN):
            _stack(SequenceType.VIDEO_LEN),
        str(SequenceType.VIDEO_NAME): [
            b[SequenceType.VIDEO_NAME] for b in batch
        ],
    }


class BboxDataset(Dataset):

  """A dataset for working with object bounding boxes."""

  def __init__(
    self,
    root_dir,
    frame_sampler,
    pusher = False,
    augmentor = None, augment = False,
    max_vids_per_class = -1,
    seed = None
  ):

    super().__init__()

    self._root_dir = root_dir
    self._frame_sampler = frame_sampler
    self._seed = seed
    self._totensor = ToTensor_bbox()
    self.pusher = pusher
    self.augment = augment

    self._available_dirs = get_subdirs(self._root_dir, nonempty=True, sort=False)

    if len(self._available_dirs) == 0:
      raise ValueError("{} is an empty directory.".format(root_dir))
    self._allowed_dirs = self._available_dirs

    self.seed_rng()
    self._build_dir_tree()

    
  def seed_rng(self):
    if self._seed:
      random.seed(self._seed)

  def restrict_subdirs(self, subdirs):

    if not isinstance(subdirs, (tuple, list)):
      subdirs = [subdirs]
    if subdirs:
      len_init = len(self._available_dirs)
      self._allowed_dirs = self._available_dirs
      subdirs = [osp.join(self._root_dir, x) for x in subdirs]
      self._allowed_dirs = list(set(self._allowed_dirs) & set(subdirs))
      if len(self._allowed_dirs) == 0:  # pylint: disable=g-explicit-length-test
        raise ValueError(f"Filtering with {subdirs} returns an empty dataset.")
      len_final = len(self._allowed_dirs)
      logging.debug(  # pylint: disable=logging-format-interpolation
          f"Restricted dataset from {len_init} to {len_final} actions.")
      self._build_dir_tree()
    else:
      logging.debug("Passed in an empty list. No action taken.")

  def _build_dir_tree(self):

    self._dir_tree = {}
    for path in self._allowed_dirs:
      vids = get_subdirs(
          path,
          nonempty=False,
          sort=True,
          sortfunc=lambda x: int(osp.splitext(osp.basename(x))[0]),
      )
      if len(vids) > 0:  # pylint: disable=g-explicit-length-test
        self._dir_tree[path] = vids

  def _get_video_path(self, class_idx, vid_idx):

      action_class = list(self._dir_tree)[class_idx]
      return self._dir_tree[action_class][vid_idx]

  def _get_data(self, vid_path):
      
      sample = self._frame_sampler.sample(vid_path)
      if self.pusher:
        frames = np.stack([load_bboxes_pusher(f) for f in sample["frames"]])
        invert = bool(np.random.choice(2, p=[0.5, 0.5]))
        scale = np.random.uniform(0.8, 1.1)
        shift = np.random.uniform(0.8, 1.1)
        if self.augment:
          transformed_frames = self.transform(bboxes=frames, shift=shift, scale=scale, invert=invert)
      else:
        frames = np.stack([load_bboxes(f) for f in sample["frames"]])

      frames = np.take(frames, sample["ctx_idxs"], axis=0)
      frames = np.reshape(frames, (-1, *frames.shape[2:]))
    
      frame_idxs = np.asarray(sample["frame_idxs"], dtype=np.int64)

      return {
          SequenceType.FRAMES: frames,
          SequenceType.FRAME_IDXS: frame_idxs,
          SequenceType.VIDEO_NAME: vid_path,
          SequenceType.VIDEO_LEN: sample["vid_len"],
      }

  def transform(self, bboxes, shift, scale, invert):
    
    transformed_bboxes = np.zeros_like(bboxes)
    
    for i in range(bboxes.shape[0]):

      if invert:
              bboxes[i]= 1 - bboxes[i]
      for obj_index in range(3):  

          x_centre = (bboxes[i][0][obj_index][2] - bboxes[i][0][obj_index][0])/2 + bboxes[i][0][obj_index][0] 
          y_centre = (bboxes[i][0][obj_index][3] - bboxes[i][0][obj_index][1])/2 + bboxes[i][0][obj_index][1]
          width =  bboxes[i][0][obj_index][2] - bboxes[i][0][obj_index][0]
          height = bboxes[i][0][obj_index][3] - bboxes[i][0][obj_index][1]

          x_centre, y_centre = shift * x_centre, shift * y_centre
          width, height = scale * width, scale * height

          transformed_bboxes[i][0][obj_index][0] = x_centre - width/2
          transformed_bboxes[i][0][obj_index][1] = y_centre - height/2
          transformed_bboxes[i][0][obj_index][2] = x_centre + width/2
          transformed_bboxes[i][0][obj_index][3] = y_centre + height/2

      
    if np.max(transformed_bboxes) > 1:
        transformed_bboxes = transformed_bboxes - (np.max(transformed_bboxes) - 1)
        
    return transformed_bboxes

  
  def __getitem__(self, idxs):

    vid_paths = self._get_video_path(*idxs)
    data_np = self._get_data(vid_paths)
    data_tensor = self._totensor(data_np)
    return data_tensor

  
  def __len__(self):

    return self.total_vids

  def num_classes(self):
    """The number of subdirs, i.e. allowed video classes."""
    return len(self._allowed_dirs)

  @property
  def total_vids(self):
    """The total number of videos across all allowed video classes."""
    num_vids = 0
    for vids in self._dir_tree.values():
      num_vids += len(vids)
    return num_vids

  @property
  def dir_tree(self):
    """The directory tree."""
    return self._dir_tree

  def collate_fn(
      self,
      batch,
  ):
    """A custom collate function for video data."""

    def _stack(key):
      return torch.stack([b[key] for b in batch])

    # Convert the keys to their string representation so that a batch can be
    # more easily indexed into without an extra import to SequenceType.
    return {
        str(SequenceType.FRAMES):
            _stack(SequenceType.FRAMES),
        str(SequenceType.FRAME_IDXS):
            _stack(SequenceType.FRAME_IDXS),
        str(SequenceType.VIDEO_LEN):
            _stack(SequenceType.VIDEO_LEN),
        str(SequenceType.VIDEO_NAME): [
            b[SequenceType.VIDEO_NAME] for b in batch
        ],
    }



class BboxCombinedDataset(Dataset):

  """A dataset for working with object bounding boxes."""

  def __init__(
    self,
    root_dir,
    frame_sampler,
    pusher = False,
    augmentor = None, augment = True,
    max_vids_per_class = -1,
    seed = None, box_prefix = "bboxes", img_prefix = "images", history_frames = 1
  ):

    super().__init__()

    self._root_dir = root_dir
    self._frame_sampler = frame_sampler
    self._seed = seed
    self._totensor = ToTensor_bbox()
    self.pusher = pusher
    self.augment = augment
    self.box_prefix = box_prefix
    self.img_prefix = img_prefix
    self.count = 0
    self.history_frames = history_frames

    self._available_dirs = get_subdirs(self._root_dir + f"/{box_prefix}", nonempty=True, sort=False)

    if len(self._available_dirs) == 0:
      raise ValueError("{} is an empty directory.".format(root_dir))
    self._allowed_dirs = self._available_dirs

    self.seed_rng()
    self._build_dir_tree()

    
  def seed_rng(self):
    if self._seed:
      random.seed(self._seed)

  def restrict_subdirs(self, subdirs):

    if not isinstance(subdirs, (tuple, list)):
      subdirs = [subdirs]
    if subdirs:
      len_init = len(self._available_dirs)
      self._allowed_dirs = self._available_dirs
      subdirs = [osp.join(self._root_dir, self.box_prefix, x) for x in subdirs]
      self._allowed_dirs = list(set(self._allowed_dirs) & set(subdirs))
      if len(self._allowed_dirs) == 0:  # pylint: disable=g-explicit-length-test
        raise ValueError(f"Filtering with {subdirs} returns an empty dataset.")
      len_final = len(self._allowed_dirs)
      logging.debug(  # pylint: disable=logging-format-interpolation
          f"Restricted dataset from {len_init} to {len_final} actions.")
      self._build_dir_tree()
    else:
      logging.debug("Passed in an empty list. No action taken.")

  def _build_dir_tree(self):

    self._dir_tree = {}
    for path in self._allowed_dirs:
      vids = get_subdirs(
          path,
          nonempty=False,
          sort=True,
          sortfunc=lambda x: int(osp.splitext(osp.basename(x))[0]),
      )
      if len(vids) > 0:  # pylint: disable=g-explicit-length-test
        self._dir_tree[path] = vids

  def _get_video_path(self, class_idx, vid_idx):

      action_class = list(self._dir_tree)[class_idx]
      return self._dir_tree[action_class][vid_idx]

  def _get_data(self, vid_path):
      
      sample = self._frame_sampler.sample(vid_path)

      images = np.stack([np.ones((224, 224, 3)) for f in sample["frames"]] )  #np.stack([load_image_bbox(f, self.img_prefix) for f in sample["frames"]])
      images = np.repeat(images, self.history_frames, axis=0)
      if self.pusher:

        bbox_frames = np.stack([load_bboxes_pusher_dists_context(f, self.history_frames) for f in sample["frames"]])
      
      else:
        bbox_frames = np.stack([load_bboxes(f) for f in sample["frames"]])

      bbox_shape = bbox_frames.shape  
      if self.augment:
        
        bbox_frames = np.reshape(bbox_frames, (bbox_shape[0]*bbox_shape[1], 1, bbox_shape[2], bbox_shape[3])) #flattening out the history frames
        
        #bbox_frames = (np.random.rand(1)*0.6 - 0.3) + bbox_frames
        
        bbox_frames = self.transform(bbox_frames, images)
        bbox_frames = np.reshape(bbox_frames, (bbox_shape)) #-1 x history_frames x num_objects x feature_dims
      
      bbox_frames = np.transpose(bbox_frames, (0, 2, 1, 3)) #-1 x num_objects x history_frames x feature_dims
      bbox_frames = np.reshape(bbox_frames, (bbox_shape[0], 1, bbox_shape[2], bbox_shape[1] * bbox_shape[3]))  # -1 x 1 x num_objects x (history_frames x feature_dims)
      bbox_frames = np.take(bbox_frames, sample["ctx_idxs"], axis=0)
      bbox_frames = np.reshape(bbox_frames, (-1, *bbox_frames.shape[2:]))
    
      img_frames = np.take(images, sample["ctx_idxs"], axis=0)
      img_frames = np.reshape(img_frames, (-1, *img_frames.shape[2:]))
    
      frame_idxs = np.asarray(sample["frame_idxs"], dtype=np.int64)
      
      return {
          SequenceType.FRAMES: bbox_frames,
          SequenceType.FRAME_IDXS: frame_idxs,
          SequenceType.VIDEO_NAME: vid_path,
          SequenceType.VIDEO_LEN: sample["vid_len"],
      }

  def transform(self, bboxes, images):
    
    b, t, w, h = bboxes.shape
    bboxes = np.reshape(bboxes, (b, t*w, h))
    
    transforms = self.create_transform_sequence()
    transformed_bboxes = np.zeros_like(bboxes)
    
    img_h, img_w = images.shape[1:3]

    for i, (img, box) in enumerate(zip(images, bboxes)):

      box= self.unnormalize_bbox(box, (img_h, img_w))

      _, transformed_bbox = transforms(img, box)
      transformed_bboxes[i] = self.normalize_bbox(transformed_bbox, (img_w, img_h))

    return np.reshape(transformed_bboxes, (b, t, w, h))

  
  def test_plot(self, image, bboxes):
    if self.count < 25:
 
      img_h, img_w = 224, 224
      canvas = np.ones((224, 244, 3)) * 255

      for bbox in bboxes:

        img = cv2.rectangle(canvas, (int(bbox[0]*img_w), int(bbox[1]*img_h)),(int(bbox[2]*img_w), int(bbox[3]*img_h)), (255, 0, 0), 1)

      cv2.imwrite(f"test_aug_{self.count}.png", img)  
      self.count += 1
  def __getitem__(self, idxs):

    vid_paths = self._get_video_path(*idxs)
    data_np = self._get_data(vid_paths)
    data_tensor = self._totensor(data_np)
    return data_tensor

  
  def __len__(self):

    return self.total_vids

  def normalize_bbox(self, bboxes, img_shape):

    normalized_bboxes = np.zeros_like(bboxes)

    for i, bbox in enumerate(bboxes):
      
      normalized_bboxes[i] = bbox/img_shape[0] #assumes image is a square
    return normalized_bboxes 

  def unnormalize_bbox(self, bboxes, img_shape):

    unnormalized_bboxes = np.zeros_like(bboxes)

    for i, bbox in enumerate(bboxes):
      
      unnormalized_bboxes[i] = bbox*img_shape[0] #assumes image is a square 
    return unnormalized_bboxes

  def num_classes(self):
    """The number of subdirs, i.e. allowed video classes."""
    return len(self._allowed_dirs)

  @property
  def total_vids(self):
    """The total number of videos across all allowed video classes."""
    num_vids = 0
    for vids in self._dir_tree.values():
      num_vids += len(vids)
    return num_vids

  @property
  def dir_tree(self):
    """The directory tree."""
    return self._dir_tree

  def collate_fn(
      self,
      batch,
  ):
    """A custom collate function for video data."""

    def _stack(key):
      return torch.stack([b[key] for b in batch])

    # Convert the keys to their string representation so that a batch can be
    # more easily indexed into without an extra import to SequenceType.
    return {
        str(SequenceType.FRAMES):
            _stack(SequenceType.FRAMES),
        str(SequenceType.FRAME_IDXS):
            _stack(SequenceType.FRAME_IDXS),
        str(SequenceType.VIDEO_LEN):
            _stack(SequenceType.VIDEO_LEN),
        str(SequenceType.VIDEO_NAME): [
            b[SequenceType.VIDEO_NAME] for b in batch
        ],
    }

  def create_transform_sequence(self):

    vflip = np.random.choice(2, p=(0.7, 0.3))
    hflip = np.random.choice(2, p=(0.9, 0.1))
    scale, c_scale = np.random.uniform(-0.2, 0.2), np.random.choice(2, p=[0.9, 0.1])
    rotate, c_rotate = np.random.uniform(0, 45), np.random.choice(2, p=[0.9, 0.1])
    shift = np.random.choice(2, p=(0.3, 0.7))
    
    transforms = []
    if vflip:
        transforms.append(VerticalFlip())

    # if hflip:
    #     transforms.append(HorizontalFlip())
    
    # if c_scale:
    #     transforms.append(Scale(scale))
    
    # if c_rotate:
    #     transforms.append(Rotate(rotate))

    if shift:
       transforms.append(Shift())
    
    

    transforms = Sequence(transforms)

    return transforms


  



  
  


if __name__ == "__main__":


  frame_sampler = frame_samplers.UniformSampler(num_frames = 50, num_ctx_frames = 1, ctx_stride = 1, pattern = "*.txt", offset = 0)
  #print(load_bboxes("/home/xiaolonw/sateesh/object_graph/data/data_xirl_bboxes/train/gripper/0000/000026.txt"))
  
  dataset = BboxCombinedDataset("/shared/xiaolonw/sateesh/object_graph/data/data_mujoco_combined_test/train", frame_sampler, pusher=True)

  print(dataset.__getitem__([0, 0])[SequenceType.FRAMES].shape)