import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from src.config import Config
from src.edge_detect import EdgeDetect
from src.utils import create_dir


def main(mode, config_path):
    r"""
    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    config = load_config(mode, config_path)
    config.CONFIG_PATH = config_path

    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # build the model and initialize
    model = EdgeDetect(config)
    model.load()

    # model pre training
    if config.MODE == 0:
        config.print()
        print('\nstart pre_training...\n')
        model.train()

    # model training
    if config.MODE == 1:
        config.print()
        print('\nstart training...\n')
        model.train()

    # model test
    elif config.MODE == 2:
        print('\nstart testing...\n')
        model.test()

    # model eval on val set
    elif config.MODE == 3:
        print('\nstart eval...\n')
        model.eval()

    # model eval on test set
    elif config.MODE == 4:
        print('\nstart eval...\n')
        model.eval()


def load_config(mode, config_path):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test 3:eval reads from config file if not specified
    """

    # load config file
    config = Config(config_path)

    # pre train mode
    if mode == 0:
        config.MODE = 0

    # train mode
    if mode == 1:
        config.MODE = 1

    # test mode
    elif mode == 2:
        config.MODE = 2

    # eval mode
    elif mode == 3:
        config.MODE = 3
    elif mode == 4:
        config.MODE = 4
    return config


if __name__ == "__main__":
    main()
