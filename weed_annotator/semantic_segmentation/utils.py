import json
from logging import getLogger
import os
import random
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import wandb
from weed_annotator.semantic_segmentation.logger import create_logger

logger = getLogger()

def set_seeds(seed=43):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    # Set to True if training should run faster. Then reproducabiliy is not guaranteed anymore.
    cudnn.benchmark = False

def load_img_list(subset_file):
    data_path = os.path.dirname(subset_file)
    with open(f'{subset_file}', 'r+') as f:
        img_list = [line.replace('\n', '') for line in f.readlines()]
        img_list = [f"{data_path}/{img_file}" for img_file in img_list]
    return img_list

def init_logging(config):
    log_path = f"{config['logging']['log_path']}/{config['identifier']}"
    os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, "params.json"), "w") as f:
        json.dump(config, f)

    #checkpoints
    dump_checkpoints = f"{log_path}/checkpoints"
    os.makedirs(dump_checkpoints, exist_ok=True)

    # logging
    logger = create_logger(os.path.join(log_path, "train.log"))

    # tensorboar logging
    writer = None
    if config["logging"]["tool"] == "wandb":
        wandb.init(project=config["logging"]["project_name"], name=config["identifier"], config=config, dir=log_path)
    else:
        tb_log_path = f"{log_path}/tb"
        os.makedirs(tb_log_path, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_log_path)

    logger.info("The experiment will be stored in %s\n" % log_path)

    return logger, writer, dump_checkpoints

class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def restart_from_checkpoint(ckp_paths, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    # look for a checkpoint in exp repository
    if isinstance(ckp_paths, list):
        for ckp_path in ckp_paths:
            if os.path.isfile(ckp_path):
                break
    else:
        ckp_path = ckp_paths

    if not os.path.isfile(ckp_path):
        return

    logger.info("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path)

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(msg)
            except TypeError:
                msg = value.load_state_dict(checkpoint[key])
            logger.info("=> loaded {} from checkpoint '{}'".format(key, ckp_path))
        else:
            logger.warning(
                "=> failed to load {} from checkpoint '{}'".format(key, ckp_path)
            )

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


