import random
import torch
import numpy as np

def set_seeds(seed=43):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)