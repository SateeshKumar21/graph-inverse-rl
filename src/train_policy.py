'''
learn a policy w/ learned reward
'''

from absl import app
from absl import flags
from absl import logging as logger
from configs.constants import *
from ml_collections.config_flags import DEFINE_config_file
import logging


FLAGS = flags.FLAGS

flags.DEFINE_string("device", "cuda:0", "The compute device.")

formatter = logging.Formatter(
        fmt="[%(levelname)s] [%(asctime)s] [%(module)s.py:%(lineno)s] %(message)s",
        datefmt="%b-%d-%y %H:%M:%S"
    )

logger.get_absl_handler().setFormatter(formatter)

def main(_):

    config = FLAGS.config


if __name__ == "__main__":
    app.run(main)