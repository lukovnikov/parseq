import torch
import numpy as np
import re
from tqdm import tqdm

from parseq.datasets import CFQDatasetLoader


def try_cfq():
    ds = CFQDatasetLoader().load("random/modent")
    print(ds[0])


if __name__ == '__main__':
    try_cfq()