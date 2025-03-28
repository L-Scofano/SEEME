#from .base import BASEDataModule
from os.path import join as pjoin
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .humanml.data.dataset import Text2MotionDatasetV2, TextOnlyDataset, EgoBodyData, DatasetEgobody, EgoBodyData2, EgoBodyData3, GimoData
import torch

class BASEDataModule(pl.LightningDataModule):

    def __init__(self, collate_fn, batch_size: int, num_workers: int):
        super().__init__()

        # self.dataloader_options = {
        #     "batch_size": batch_size, "num_workers": num_workers,"collate_fn": collate_datastruct_and_text}
        self.dataloader_options = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "collate_fn": collate_fn,
        }

        # self.collate_fn = collate_fn
        self.persistent_workers = True
        self.is_mm = False

    def get_sample_set(self, overrides={}):
        sample_params = self.hparams.copy()
        sample_params.update(overrides)
        #split_file = pjoin(
        #    eval(f"self.cfg.DATASET.{self.name.upper()}.SPLIT_ROOT"),
        #    'val' + ".txt",
        #)
        split_file = pjoin(
            eval(f"self.cfg.DATASET.{self.name.upper()}.SPLIT_ROOT"),
            'val' + ".txt",
        )

        #dt_file = 'annotation_egocentric_smpl_npz/egocapture_train_smpl.npz'
        
        #return self.Dataset(**sample_params)
        return self.Dataset(split_file=split_file, **sample_params)

    def __getattr__(self, item):
        # train_dataset/val_dataset etc cached like properties
        if item.endswith("_dataset") and not item.startswith("_"):
            subset = item[:-len("_dataset")]
            item_c = "_" + item
            if item_c not in self.__dict__:
                subset = subset.upper() if subset != "val" else "EVAL"
                split = eval(f"self.cfg.{subset}.SPLIT")
                split_file = pjoin(
                    eval(f"self.cfg.DATASET.{self.name.upper()}.SPLIT_ROOT"),
                    eval(f"self.cfg.{subset}.SPLIT") + ".txt",
                )
                self.__dict__[item_c] = self.Dataset(split_file=split_file,
                                                     split=split,
                                                     **self.hparams)
            return getattr(self, item_c)
        classname = self.__class__.__name__
        raise AttributeError(f"'{classname}' object has no attribute '{item}'")
    
    def setup(self, stage=None):
        self.stage = stage
        if stage == "fit" or stage is None:
            _ = self.train_dataset
            _ = self.val_dataset
        if stage == "test" or stage is None:
            _ = self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True,
            persistent_workers=True, **self.dataloader_options)
    
    def val_dataloader(self):
        dataloader_options = self.dataloader_options.copy()
        dataloader_options["shuffle"] = False
        dataloader_options['num_workers'] = self.cfg.EVAL.NUM_WORKERS
        dataloader_options["batch_size"] = self.cfg.EVAL.BATCH_SIZE
        return DataLoader(self.val_dataset, 
                          persistent_workers=True,
                          **dataloader_options)
    
    def test_dataloader(self):
        # overrides batch_size and num_workers
        dataloader_options = self.dataloader_options.copy()
        dataloader_options[
            "batch_size"] = 1 if self.is_mm else self.cfg.TEST.BATCH_SIZE
        dataloader_options["num_workers"] = self.cfg.TEST.NUM_WORKERS
        # dataloader_options["drop_last"] = True
        dataloader_options["shuffle"] = False
        return DataLoader(
            self.test_dataset,
            persistent_workers=True,
            **dataloader_options,
        )
    

class GimoDataModule(BASEDataModule):

    def __init__(self,
                 cfg,
                 batch_size,
                 num_workers,
                 collate_fn=None,
                 phase="train",
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=collate_fn)
        self.save_hyperparameters(logger=False)

        self.data_type = cfg.DATA_TYPE
        self.predict_transl = cfg.TRAIN.ABLATION.PREDICT_TRANSL


        if self.data_type=='angle':
            self.mean = np.load('./datasets/GIMO/processed/mean.npy') #mean
            self.std = np.load('./datasets/GIMO/processed/std.npy') #std
            self.numdims = 69 if self.predict_transl else 66 # ? What should self.predict_transl be?
        elif self.data_type=='rot6d':
            self.mean = np.load('./datasets/GIMO/processed/mean_rot6d.npy') #mean
            self.std = np.load('./datasets/GIMO/processed/std_rot6d.npy') #std
            self.numdims = 132

        self.name = "gimo"
        self.njoints = 21
        self.Dataset = GimoData #DatasetEgobody #EgoBodyData

        self.cfg = cfg
        sample_overrides = {
            "split": "val",
            "tiny": True,
            "progress_bar": False
        }
        self._sample_set = self.get_sample_set(overrides=sample_overrides)
        # Get additional info of the dataset
        self.nfeats = self._sample_set.nfeats # body_pose, betas, global_orient, transl

    def renorm(self, features):

    
        features = features * torch.tensor(self.std[0,:self.numdims]).to(features.device) + torch.tensor(self.mean[0,:self.numdims]).to(features.device)
        #features = features * torch.tensor(self.std[:144]).to(features.device) + torch.tensor(self.mean[:144]).to(features.device)
        
        return features
    
    # def feats2joints(self, features):
    #     mean = torch.tensor(self.hparams.mean).to(features)
    #     std = torch.tensor(self.hparams.std).to(features)
    #     features = features * std + mean
    #     return recover_from_ric(features, self.njoints)
    

