import argparse
import json
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from rich import get_console
from rich.table import Table

from mld.callback import ProgressLogger
from mld.config import parse_args
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model
from mld.utils.logger import create_logger


def main():
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.FOLDER = cfg.TEST.FOLDER
    # create logger
    logger = create_logger(cfg, phase="test")
    output_dir = Path(
        os.path.join(
            cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME), "samples_" + cfg.TIME
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(OmegaConf.to_yaml(cfg))

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)
    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        #     str(x) for x in cfg.DEVICE)
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # create dataset
    datasets = get_datasets(cfg, logger=logger, phase="test")[0]
    logger.info("datasets module {} initialized".format("".join(cfg.TRAIN.DATASETS)))

    # load npy files
    # TODO find a way to extend an already existing parser
    FEATURES_DIR = "vis_vae/gt"
    features_l = []
    for f in os.listdir(FEATURES_DIR):
        if f.endswith(".npy"):
            features = np.load(os.path.join(FEATURES_DIR, f))
            features = torch.from_numpy(features).float()
            feats2joints = datasets.feats2joints(features)
            features_l.append(features)
    features = np.stack(features_l, axis=0)
    pass


if __name__ == "__main__":
    main()
