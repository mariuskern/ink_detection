import os
import random

import numpy as np
import torch


def seed_everything(seed: int):    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    # if workers:
    #     os.environ["PL_GLOBAL_SEED"] = str(seed)
    #     os.environ["PL_SEED_WORKERS"] = "1"


# def create_generator(seed: int) -> torch.Generator:
#     generator = torch.Generator()
#     generator.manual_seed(seed)
#     return generator
