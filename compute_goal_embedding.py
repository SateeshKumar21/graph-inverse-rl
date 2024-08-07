"""Compute and store the mean goal embedding using a trained model."""

import os
import pickle
import typing

from absl import app
from absl import flags
import numpy as np
import torch
from torchkit import checkpoint
from graphirl import common
import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string("experiment_path", None, "Path to model checkpoint.")
flags.DEFINE_boolean("restore_checkpoint", True, "Restore model checkpoint.")

flags.mark_flag_as_required("experiment_path")

ModelType = torch.nn.Module
DataLoaderType = typing.Dict[str, torch.utils.data.DataLoader]


def embed(
    model,
    downstream_loader,
    device,
):
  """Embed the stored trajectories and compute mean goal embedding."""
  goal_embs = []
  init_embs = []
  for class_name, class_loader in downstream_loader.items():
    print(f"\tEmbedding class: {class_name}...")
    for batch_idx, batch in enumerate(class_loader):

      #if batch_idx % 100 == 0:
      
      out = model.infer(batch["frames"].to(device))
      emb = out.numpy().embs
      goal_embs.append(emb[-1, :])
      init_embs.append(emb[0, :])

  goal_emb = np.mean(np.stack(goal_embs, axis=0), axis=0, keepdims=True)
  dist_to_goal = np.linalg.norm(
      np.stack(init_embs, axis=0) - goal_emb, axis=-1).mean()
  distance_scale = 1.0 / dist_to_goal
  return goal_emb, distance_scale


def setup(device):
  """Load the latest embedder checkpoint and dataloaders."""
  config = common.load_config_from_dir(FLAGS.experiment_path)
  model = common.get_model(config)
  downstream_loaders = common.get_downstream_dataloaders(config, False)["train"]
  checkpoint_dir = os.path.join(FLAGS.experiment_path, "checkpoints")
  if FLAGS.restore_checkpoint:
    checkpoint_manager = checkpoint.CheckpointManager(
      checkpoint.Checkpoint(model=model),
      checkpoint_dir,
      device,
  )
    global_step = checkpoint_manager.restore_or_initialize()
    print(f"Restored model from checkpoint {global_step}.")
  else:
    print("Skipping checkpoint restore.")
  return model, downstream_loaders


def main(_):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model, downstream_loader = setup(device)
  model.to(device).eval()
  goal_emb, distance_scale = embed(model, downstream_loader, device)
  with open(os.path.join(FLAGS.experiment_path, "goal_emb.pkl"), "wb") as fp:
    pickle.dump(goal_emb, fp)
  with open(os.path.join(FLAGS.experiment_path, "distance_scale.pkl"), "wb") as fp:
    pickle.dump(distance_scale, fp)


if __name__ == "__main__":
  app.run(main)
