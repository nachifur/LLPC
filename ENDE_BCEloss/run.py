import numpy as np
from main import main
import multiprocessing
import os
from src.utils import create_config, create_dir, init_config
import yaml
from shutil import copyfile

debug = False
if __name__ == '__main__':
    # inital
    multiprocessing.set_start_method('spawn')
    checkpoints_path = './checkpoints/cell'  # model checkpoints path
    create_dir(checkpoints_path)
    config_path = os.path.join(checkpoints_path, 'config.yml')
    create_config(config_path)
    init_config(checkpoints_path, debug, EPOCH=50, INTERVAL=1000)

    # pre_train
    main(0, config_path)
    # train config
    create_config(config_path)
    init_config(checkpoints_path, debug, EPOCH=50,
                INTERVAL=1000, EVAL_INTERVAL_EPOCH=0.1)
    copyfile('checkpoints/cell/EdgeDetect_pre.pth',
             'checkpoints/cell/EdgeDetect.pth')
    # # train
    main(1, config_path)

    # test
    main(2, config_path)

    # # eval on val set
    # main(3,config_path)
    # # eval on test set
    main(4, config_path)

# tensorboard --logdir runs/edge_detect
