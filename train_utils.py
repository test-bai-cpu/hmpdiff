import random
import torch
import numpy as np


def set_random_seed(seed):
    seed = seed if seed >= 0 else random.randint(0, 2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed