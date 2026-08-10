"""Utilidades: seed, logger, checkpoints."""
import json
import logging
import os
import random
import shutil
import sys

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_logger(name: str = 'train') -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', '%H:%M:%S'))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def save_checkpoint(state: dict, out_dir: str, filename: str = 'best.pt'):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(state, os.path.join(out_dir, filename))


def save_metrics(metrics: dict, out_dir: str, filename: str = 'metrics.json'):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, filename), 'w') as f:
        json.dump(metrics, f, indent=2)


def copy_config(cfg_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(cfg_path, os.path.join(out_dir, os.path.basename(cfg_path)))
