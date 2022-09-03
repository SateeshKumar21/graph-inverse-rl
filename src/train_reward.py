'''
learn a reward function
'''
from absl import app
from absl import flags
from absl import logging as logger
from configs.constants import *
from ml_collections.config_flags import DEFINE_config_file
import logging


FLAGS = flags.FLAGS

flags.DEFINE_enum("embodiment", None, EMBODIMENTS, "Which embodiment to train. Will train all sequentially if not specified.")
flags.DEFINE_enum("algo", None, ALGORITHMS, "The pretraining algorithm to use.")
flags.DEFINE_enum("dataset", None, DATASETS, "The pretraining dataset to use.")
flags.DEFINE_string("device", "cuda:0", "The compute device.")

formatter = logging.Formatter(
        fmt="[%(levelname)s] [%(asctime)s] [%(module)s.py:%(lineno)s] %(message)s",
        datefmt="%b-%d-%y %H:%M:%S"
    )

logger.get_absl_handler().setFormatter(formatter)

def main(_):
    DEFINE_config_file("config", ALGO_TO_CONFIG[FLAGS.algo], "File path to the training hyperparameter configuration.")

    config = FLAGS.config
    config.data.root = f"datasets/{FLAGS.dataset}/"
    logger.info(f"Dataset Path: {config.data.root}")


if __name__ == "__main__":
    app.run(main)