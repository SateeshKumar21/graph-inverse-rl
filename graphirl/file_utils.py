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

"""File utilities."""

from fileinput import filename
from glob import glob  # pylint: disable=g-importing-member
import os
import os.path as osp
from typing import Callable, List, Optional, Tuple, cast

import numpy as np
from PIL import Image
from sklearn.metrics import pairwise_distances
from natsort import natsorted

def get_subdirs(
    d,
    nonempty = False,
    sort = True,
    basename = False,
    sortfunc = None,  # pylint: disable=g-bare-generic
):
  """Return a list of subdirectories in a given directory.

  Args:
    d: The path to the directory.
    nonempty: Only return non-empty subdirs.
    sort: Whether to sort in lexicographical order.
    basename: Only return the tail of the subdir paths.
    sortfunc : An optional sorting Callable to override.

  Returns:
    The list of subdirectories.
  """


  subdirs = [
      cast(str, f.path) for f in os.scandir(d) if f.is_dir()
      if not f.name.startswith(".")
  ]

  if nonempty:
    subdirs = [f for f in subdirs if not is_folder_empty(f)]
  if sort:
    if sortfunc is None:
      subdirs = natsorted(subdirs)
      #subdirs.sort(key=lambda x: osp.basename(x))  # pylint: disable=unnecessary-lambda
    else:
      subdirs = natsorted(subdirs)
  if basename:
    return [osp.basename(x) for x in subdirs]
  return subdirs


def get_files(
    d,
    pattern,
    sort = False,
    basename = False,
    sortfunc = None,  # pylint: disable=g-bare-generic
):
  """Return a list of files in a given directory.

  Args:
    d: The path to the directory.
    pattern: The wildcard to filter files with.
    sort: Whether to sort in lexicographical order.
    basename: Only return the tail of the subdir paths.
    sortfunc : An optional sorting Callable to override.

  Returns:
    The files in the directory.
  """
  files = glob(osp.join(d, pattern))
  files = [f for f in files if osp.isfile(f)]
  if sort:
    if sortfunc is None:
      files.sort(key=lambda x: osp.basename(x))  # pylint: disable=unnecessary-lambda
    else:
      files.sort(key=sortfunc)
  if basename:
    return [osp.basename(x) for x in files]
  return files


def is_folder_empty(d):
  """A folder is not empty if it contains >=1 non hidden files."""
  return len(glob(osp.join(d, "*"))) == 0  # pylint: disable=g-explicit-length-test


def load_image(
    filename,
    resize = None,
):
  """Loads an image as a numpy array.

  Args:
    filename: The name of the image file.
    resize: The height and width of the loaded image. Set to `None` to keep
      original image dims.

  Returns:
    A numpy uint8 array.
  """
  img = Image.open(filename)
  if resize is not None:
    # PIL expects a (width, height) tuple.
    img = img.resize((resize[1], resize[0]))
  return np.asarray(img)

def load_image_bbox(
    filename, img_prefix,
    resize = None,
):
  """Loads an image as a numpy array.

  Args:
    filename: The name of the image file.
    resize: The height and width of the loaded image. Set to `None` to keep
      original image dims.

  Returns:
    A numpy uint8 array.
  """
  splited_filename = filename.split("/")
  

  filename_images = os.path.join("/".join(splited_filename[:-4]), img_prefix, splited_filename[-3], os.path.basename(os.path.dirname(filename))[:-4], os.path.basename(filename)[:-4] + ".png")
  img = Image.open(filename_images)
  if resize is not None:
    # PIL expects a (width, height) tuple.
    img = img.resize((resize[1], resize[0]))
  return np.asarray(img)



def parse_bbox(frame):
  """Parses bbox given opened file.

  """
  bbox_lines = frame.readlines()
  bboxes = []
  for line in bbox_lines:

    line = line.strip().split(" ")
    bbox = np.zeros((5))

    bbox[0] = int(float(line[0]))  #label
    bbox[1] = float(line[1]) 
    bbox[2] = float(line[2])
    bbox[3] = float(line[3])
    bbox[4] = float(line[4])

    bboxes.append(list(bbox))
  
  return bboxes

def load_bboxes_pusher(f):
  bbox = np.zeros((3, 4))
  with open (f, "r") as fp:
    bboxes = parse_bbox(fp)
  for i, bbox_f in enumerate(bboxes):
    bbox[i] = bbox_f[1:]

  return np.reshape(bbox, (1, 3, 4))

def load_bboxes_pusher_dists(f):
  bbox = np.zeros((3, 4))
  with open (f, "r") as fp:
    bboxes = parse_bbox(fp)
  for i, bbox_f in enumerate(bboxes):
      bbox[i] = bbox_f[1:]

  centres_y, centres_x = (bbox[:, 3] + bbox[:, 1])/2, (bbox[:, 2] + bbox[:, 0])/2 

  centres = np.column_stack([centres_x, centres_y])
  distances = pairwise_distances(centres)
 
  final_features = np.column_stack([bbox, distances])

  return np.reshape(final_features, (1, 3, 7))


def load_bboxes_pusher_dists_context(f, num_history_frames):
  
  file_index = int(os.path.basename(f)[:-4])

  bboxes = np.zeros((num_history_frames, 3, 7))
 #5 - (2 - 1) 6
 #(4, 6)

  for i, index in enumerate(range(file_index - (num_history_frames - 1) , file_index + 1)):
    
    curr_file_index = max(0, index)
    f = os.path.join(os.path.dirname(f), str(curr_file_index).zfill(6) + ".txt")
    bbox = np.zeros((3, 4))
    with open (f, "r") as fp:
      bboxes_files = parse_bbox(fp)
      
    for j, bbox_f in enumerate(bboxes_files):
        bbox[j] = bbox_f[1:]

    centres_y, centres_x = (bbox[:, 3] + bbox[:, 1])/2, (bbox[ :, 2] + bbox[ :, 0])/2 

    centres = np.column_stack([centres_x, centres_y])
    distances = pairwise_distances(centres)
  
    bboxes[i] = np.column_stack([bbox, distances])
    #print("Number of zeros in goalbboxes: {}")
  #print(bboxes)
  #bboxes = np.transpose(bboxes, (1, 0, 2)) s
  return np.reshape(bboxes, (num_history_frames, 3, 7))



# def load_bboxes_pusher_dists_multiple(f, context_frames=3):

#   index = int(os.path.basename(f)[:-4])
  
#   bbox = np.zeros((3, 4))
#   with open (f, "r") as fp:
#     bboxes = parse_bbox(fp)
#   for i, bbox_f in enumerate(bboxes):
#       bbox[i] = bbox_f[1:]

#   centres_y, centres_x = (bbox[:, 3] + bbox[:, 1])/2, (bbox[:, 2] + bbox[:, 0])/2 

#   centres = np.column_stack([centres_x, centres_y])
#   distances = pairwise_distances(centres)
 
#   final_features = np.column_stack([bbox, distances])

#   final_features = np.tile(final_features, (context_frames, 1))

#   for i, frame_num in enumerate(range(index, index+context_frames)):

#       file_name = os.path.join(os.path.dirname(f), str(frame_num).zfill(6) + ".txt")
#       if os.path.exists(file_name):
#         print("File exists")
#         final_features[3*(i+1):3*(i+2)] = load_bboxes_pusher_dists(file_name)[0]
    

      

#   return np.reshape(final_features, (1, 3*context_frames, 7))




def load_bboxes_pusher_dists_gp(f):
  bbox = np.zeros((2, 4))
  with open (f, "r") as fp:
    bboxes = parse_bbox(fp)
  for i, bbox_f in enumerate(bboxes):
    if i > 0:
      bbox[i-1] = bbox_f[1:]

  centres_y, centres_x = (bbox[:, 3] + bbox[:, 1])/2, (bbox[:, 2] + bbox[:, 0])/2 

  centres = np.column_stack([centres_x, centres_y])
  distances = pairwise_distances(centres)
 
  final_features = np.column_stack([bbox, distances])

  return np.reshape(final_features, (1, 2, 6))



def load_bboxes_pusher_gp(f):

    bbox = np.zeros((2, 4))
    with open (f, "r") as fp:
      bboxes = parse_bbox(fp)
    for i, bbox_f in enumerate(bboxes):
      if i > 0:
        bbox[i-1] = bbox_f[1:]

    return np.reshape(bbox, (1, 2, 4))

def load_bboxes(f):
  #print(f)
  bbox = np.zeros((4, 4))
  with open(f, "r") as fp:
    bboxes = parse_bbox(fp)
    #print(bboxes) 
  box_i = 1
  
  for bbox_f in bboxes:
    if bbox_f[0] == 0: #agent
      bbox[0] = bbox_f[1:]
    else: #box

      bbox[box_i] = bbox_f[1:]
      box_i += 1

  return np.reshape(bbox, (1, 4, 4))

def load_bboxes_ordered(f):

  bbox = np.zeros((4, 4))
  with open(f, "r") as fp:
    bboxes = parse_bbox(fp) 
  
  for bbox_f in bboxes:
    
    if bbox_f[0] == 0: #agent
      bbox[-1] = bbox_f[1:]
    elif bbox_f[0] == 2: #box1 
      bbox[1] = bbox_f[1:]
    elif bbox_f[1] == 3: #box2
      bbox[2] = bbox_f[1:]
    elif bbox_f[2] == 4: #box3
      bbox[3] = bbox_f[1:]


  return np.reshape(bbox, (1, 4, 4))


# if __name__ == "__main__":

  