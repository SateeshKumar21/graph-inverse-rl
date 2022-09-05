'''
learn a reward function
'''
from absl import app
from absl import flags
from absl import logging as logger
from base_configs import validate_config
from configs.constants import *
from ml_collections.config_flags import DEFINE_config_file
from torchkit.torchkit.utils.seed import seed_rngs, set_cudnn
from torchkit.torchkit.checkpoint import CheckpointManager
from torchkit.torchkit.utils.timer import Stopwatch
from utils import setup_experiment, load_config_from_dir
from graphirl import common
import logging
import os
import os.path as osp
import torch
import wandb
import subprocess
import yaml
import json

FLAGS = flags.FLAGS

flags.DEFINE_enum("embodiment", None, EMBODIMENTS,
                  "Which embodiment to train. Will train all sequentially if not specified.")
flags.DEFINE_enum("algo", None, ALGORITHMS,
                  "The pretraining algorithm to use.")
flags.DEFINE_enum("dataset", None, DATASETS, "The pretraining dataset to use.")
flags.DEFINE_string("device", "cuda:0", "The compute device.")

formatter = logging.Formatter(
    fmt="[%(levelname)s] [%(asctime)s] [%(module)s.py:%(lineno)s] %(message)s",
    datefmt="%b-%d-%y %H:%M:%S"
)

logger.get_absl_handler().setFormatter(formatter)


def main(_):
    DEFINE_config_file(
        "config", ALGO_TO_CONFIG[FLAGS.algo], "File path to the training hyperparameter configuration.")

    config = FLAGS.config
    config.data.root = f"datasets/{FLAGS.dataset}/"
    logger.info(f"Dataset Path: {config.data.root}")

    validate_config(config, mode="pretrain")

    exp_dir = osp.join(config.root_dir, FLAGS.experiment_name)
    setup_experiment(exp_dir, config, FLAGS.resume)

    if FLAGS.raw_imagenet:
        return

    # Setup compute device.
    if torch.cuda.is_available():
        device = torch.device(FLAGS.device)
    else:
        logging.info("No GPU device found. Falling back to CPU.")
        device = torch.device("cpu")
    logging.info("Using device: %s", device)

    # Set RNG seeds.
    if config.seed is not None:
        logging.info("Pretraining experiment seed: %d", config.seed)
        seed_rngs(config.seed)
        set_cudnn(config.cudnn_deterministic, config.cudnn_benchmark)
    else:
        logging.info(
            "No RNG seed has been set for this pretraining experiment.")

    algo = flags.FLAGS.flag_values_dict()["config"]["algorithm"]
    model_type = flags.FLAGS.flag_values_dict(
    )["config"]["model"]["model_type"]

    # weights and biases for logging
    run = wandb.init(project=f"{FLAGS.wandb_project}",
                     entity=f"{FLAGS.wandb_entity}",
                     group=f"{FLAGS.experiment_name}",
                     name=f"seed={config.seed}")

    # Load factories.
    (
        model,
        optimizer,
        pretrain_loaders,
        downstream_loaders,
        trainer,
        eval_manager,
    ) = common.get_factories(config, device)

    # Create checkpoint manager.
    checkpoint_dir = osp.join(exp_dir, "checkpoints")
    checkpoint_manager = CheckpointManager(
        checkpoint_dir,
        model=model,
        optimizer=optimizer,
    )

    global_step = checkpoint_manager.restore_or_initialize()
    total_batches = max(1, len(pretrain_loaders["train"]))
    epoch = int(global_step / total_batches)
    complete = False
    stopwatch = Stopwatch()
    try:
        while not complete:
            for batch in pretrain_loaders["train"]:
                train_loss = trainer.train_one_iter(batch)

                if not global_step % config.logging_frequency:
                    for k, v in train_loss.items():
                        wandb.log(data={f"pretrain/{k}": v}, step=global_step)

                if not global_step % config.eval.eval_frequency:
                # Evaluate the model on the pretraining validation dataset.
                    valid_loss = trainer.eval_num_iters(
                        pretrain_loaders["valid"],
                        config.eval.val_iters,
                    )
                for k, v in valid_loss.items():
                    wandb.log(data={f"pretrain/{k}": v}, step=global_step)

            # Evaluate the model on the downstream datasets.
            for split, downstream_loader in downstream_loaders.items():
                eval_to_metric = eval_manager.evaluate(
                    model,
                    downstream_loader,
                    device,
                    config.eval.val_iters,
                )
                for eval_name, eval_out in eval_to_metric.items():
                    wandb.log(data={f"downstream/{split}/{eval_name}": v}, step=global_step)

            # Exit if complete.
            global_step += 1
            if global_step > config.optim.train_max_iters:
                complete = True
                break

            time_per_iter = stopwatch.elapsed()
            logging.info(
                "Iter[{}/{}] (Epoch {}), {:.6f}s/iter, Loss: {:.3f}".format(
                    global_step,
                    config.optim.train_max_iters,
                    epoch,
                    time_per_iter,
                    train_loss["train/total_loss"].item(),
                ))
            stopwatch.reset()
        epoch += 1

    except KeyboardInterrupt:
        logging.info("Caught keyboard interrupt. Saving model before quitting.")

    finally:
        checkpoint_manager.save(global_step, run)
        # Note: This assumes that the config.root_dir value has not been
        # changed to its default value of 'tmp/xirl/pretrain_runs/'.
        exp_path = osp.join("../logs/pretraining/", FLAGS.experiment_name)
        # Dump experiment metadata as yaml file.
        with open(osp.join(exp_path, "metadata.yaml"), "w") as fp:
            yaml.dump(json.loads(FLAGS.kwargs), fp)
        
        name = osp.join(exp_path, "metadata.yaml")
        print("FILE NAME:", name)
        run.save(name, base_path=exp_path)

        # The 'goal_classifier' baseline does not need to compute a goal embedding.
        if algo != "goal_classifier":
            subprocess.run(
                [
                    "python",
                    "compute_goal_embedding.py",
                    "--experiment_path",
                    exp_path
                ],
                check=True,
            )

        # save run files
        file_names = ["goal_emb.pkl", "distance_scale.pkl", "exp_config.yaml", "git_hash.txt"]

        for name in file_names:
            filename = os.path.join(exp_path, name)
            if name == "exp_config.yaml":
                c = load_config_from_dir(exp_path)
                run.save(str(filename), base_path=exp_path)
                continue
            run.save(str(filename), base_path=exp_path)


if __name__ == "__main__":
    app.run(main)
