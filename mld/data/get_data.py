from os.path import join as pjoin

import numpy as np
from .humanml.utils.word_vectorizer import WordVectorizer
from .HumanML3D import HumanML3DDataModule
from .Kit import KitDataModule
from .Humanact12 import Humanact12DataModule
from .Uestc import UestcDataModule
from .EgoBody import EgoBodyDataModule
from .Gimo import GimoDataModule
from .utils import *


def get_mean_std(phase, cfg, dataset_name):
    # if phase == 'gt':
    #     # used by T2M models (including evaluators)
    #     mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    #     std = np.load(pjoin(opt.meta_dir, 'std.npy'))
    # elif phase in ['train', 'val', 'text_only']:
    #     # used by our models
    #     mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    #     std = np.load(pjoin(opt.data_root, 'Std.npy'))

    # todo: use different mean and val for phases
    name = "t2m" if dataset_name == "humanml3d" else dataset_name
    assert name in ["t2m", "kit"]
    # if phase in ["train", "val", "test"]:
    if phase in ["val"]:
        if name == 't2m':
            data_root = pjoin(cfg.model.t2m_path, name, "Comp_v6_KLD01",
                              "meta")
        elif name == 'kit':
            data_root = pjoin(cfg.model.t2m_path, name, "Comp_v6_KLD005",
                              "meta")
        else:
            raise ValueError("Only support t2m and kit")
        mean = np.load(pjoin(data_root, "mean.npy"))
        std = np.load(pjoin(data_root, "std.npy"))
    else:
        data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
        mean = np.load(pjoin(data_root, "Mean.npy"))
        std = np.load(pjoin(data_root, "Std.npy"))

    return mean, std


def get_WordVectorizer(cfg, phase, dataset_name):
    if phase not in ["text_only"]:
        if dataset_name.lower() in ["humanml3d", "kit"]:
            return WordVectorizer(cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")
        else:
            raise ValueError("Only support WordVectorizer for HumanML3D")
    else:
        return None


def get_collate_fn(name, phase="train"):
    if name.lower() in ["humanml3d", "kit"]:
        return mld_collate
    elif name.lower() in ["humanact12", 'uestc']:
        return a2m_collate
    # else:
    #     return all_collate
    # if phase == "test":
    #     return eval_collate
    # else:

def get_collate_ego(name):
    if name.lower() in ["egobody"]:
        return ego_collate
    else:
        return None


# map config name to module&path
dataset_module_map = {
    "humanml3d": HumanML3DDataModule,
    "kit": KitDataModule,
    "humanact12": Humanact12DataModule,
    "uestc": UestcDataModule,
}
motion_subdir = {"humanml3d": "new_joint_vecs", "kit": "new_joint_vecs"}


def get_datasets(cfg, logger=None, phase="train"):
    # get dataset names form cfg
    dataset_names = eval(f"cfg.{phase.upper()}.DATASETS")
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name.lower() in ["humanml3d", "kit"]:
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            # get mean and std corresponding to dataset
            mean, std = get_mean_std(phase, cfg, dataset_name)
            mean_eval, std_eval = get_mean_std("val", cfg, dataset_name)
            # get WordVectorizer
            wordVectorizer = get_WordVectorizer(cfg, phase, dataset_name)
            # get collect_fn
            collate_fn = get_collate_fn(dataset_name, phase)
            # get dataset module
            dataset = dataset_module_map[dataset_name.lower()](
                cfg=cfg,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                collate_fn=collate_fn,
                mean=mean,
                std=std,
                mean_eval=mean_eval,
                std_eval=std_eval,
                w_vectorizer=wordVectorizer,
                text_dir=pjoin(data_root, "texts"),
                motion_dir=pjoin(data_root, motion_subdir[dataset_name]),
                max_motion_length=cfg.DATASET.SAMPLER.MAX_LEN,
                min_motion_length=cfg.DATASET.SAMPLER.MIN_LEN,
                max_text_len=cfg.DATASET.SAMPLER.MAX_TEXT_LEN,
                unit_length=eval(
                    f"cfg.DATASET.{dataset_name.upper()}.UNIT_LEN"),
            )
            datasets.append(dataset)
        elif dataset_name.lower() in ["humanact12", 'uestc']:
            # get collect_fn
            collate_fn = get_collate_fn(dataset_name, phase)
            # get dataset module
            dataset = dataset_module_map[dataset_name.lower()](
                datapath=eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT"),
                cfg=cfg,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                collate_fn=collate_fn,
                num_frames=cfg.DATASET.HUMANACT12.NUM_FRAMES,
                sampling=cfg.DATASET.SAMPLER.SAMPLING,
                sampling_step=cfg.DATASET.SAMPLER.SAMPLING_STEP,
                pose_rep=cfg.DATASET.HUMANACT12.POSE_REP,
                max_len=cfg.DATASET.SAMPLER.MAX_LEN,
                min_len=cfg.DATASET.SAMPLER.MIN_LEN,
                num_seq_max=cfg.DATASET.SAMPLER.MAX_SQE
                if not cfg.DEBUG else 100,
                glob=cfg.DATASET.HUMANACT12.GLOB,
                translation=cfg.DATASET.HUMANACT12.TRANSLATION)
            cfg.DATASET.NCLASSES = dataset.nclasses
            datasets.append(dataset)
        elif dataset_name.lower() in ["amass"]:
            # todo: add amass dataset
            raise NotImplementedError
        
        elif dataset_name.lower() in ["egobody"]:
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")

            # NOT USE COLLATE_FN
            def collate_fn(batch):
                feats_ref, transl, beta, utils_, scene = batch[0]
                print(type(feats_ref), type(transl), type(beta), type(utils_), type(scene))
                quit()

            # get collect_fn
            #collate_fn = get_collate_ego(dataset_name)
            # get dataset module
            mean = np.load(pjoin(eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT"), "our_process_smpl_split_NEW/mean.npy"))
            std = np.load(pjoin(eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT"), "our_process_smpl_split_NEW/std.npy"))
            
            if cfg.MOTION_LENGTH != 1:
                motion_dir = pjoin(eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT"), "our_process_smpl_split_NEW")
            else:
                motion_dir = pjoin(eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT"), "our_process_smpl_split_NEW_forEgoHMR")
            

            dataset = EgoBodyDataModule(
                datapath=eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT"),
                cfg=cfg,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                #collate_fn=collate_fn,
                num_seq_max=cfg.DATASET.SAMPLER.MAX_SQE
                if not cfg.DEBUG else 100,
                mean=mean,
                std=std,
                motion_dir=motion_dir,
                condition=cfg.model.condition,
                interactee_pred=cfg.TEST.INTERACTEE_PRED,
                pred_global_orient=cfg.TEST.GLOBAL_ORIENT_PRED,
                motion_length=cfg.MOTION_LENGTH,
                data_type=cfg.DATA_TYPE,
                pred_betas=cfg.TEST.BETAS_PRED,
                transl_egoego=cfg.TEST.TRANSL_EGOEGO,
                global_orient_egoego=cfg.TEST.GLOBAL_ORIENT_EGOEGO,
                predict_transl=cfg.TRAIN.ABLATION.PREDICT_TRANSL,
                droid_slam_cut=cfg.TEST.DROID_SLAM_CUT,
                pose_estimation_task=cfg.TEST.POSE_ESTIMATION_TASK,
                )
            datasets.append(dataset)

        elif dataset_name.lower() in ["gimo"]:
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")

            # * Load mean and std
            mean = np.load(pjoin(eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT"), "processed/mean.npy")) # (1,N)
            std = np.load(pjoin(eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT"), "processed/std.npy")) # (1,N)

            motion_dir = pjoin(eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT"), "processed")
            
            dataset = GimoDataModule(
                datapath=eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT"),
                cfg=cfg,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                #collate_fn=collate_fn,
                num_seq_max=cfg.DATASET.SAMPLER.MAX_SQE
                if not cfg.DEBUG else 100,
                mean=mean,
                std=std,
                motion_dir=motion_dir,
                condition=cfg.model.condition,
                interactee_pred=cfg.TEST.INTERACTEE_PRED,
                pred_global_orient=cfg.TEST.GLOBAL_ORIENT_PRED,
                motion_length=cfg.MOTION_LENGTH,
                data_type=cfg.DATA_TYPE,
                pred_betas=cfg.TEST.BETAS_PRED,
                transl_egoego=cfg.TEST.TRANSL_EGOEGO,
                global_orient_egoego=cfg.TEST.GLOBAL_ORIENT_EGOEGO,
                predict_transl=cfg.TRAIN.ABLATION.PREDICT_TRANSL,
                droid_slam_cut=cfg.TEST.DROID_SLAM_CUT,
                pose_estimation_task=cfg.TEST.POSE_ESTIMATION_TASK,
                )
            datasets.append(dataset)
        
        elif dataset_name.lower() in ["egobody_old"]:
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")

            # get collect_fn
            #collate_fn = get_collate_ego(dataset_name)
            # get dataset module
            mean = np.load(pjoin(eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT"), "Mean.npy"))
            std = np.load(pjoin(eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT"), "Std.npy"))
            
            motion_dir = pjoin(eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT"), "our_process_smpl_1frame")
            

            dataset = EgoBodyDataModule(
                datapath=eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT"),
                cfg=cfg,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                #collate_fn=collate_fn,
                num_seq_max=cfg.DATASET.SAMPLER.MAX_SQE
                if not cfg.DEBUG else 100,
                mean=mean,
                std=std,
                motion_dir=motion_dir,
                )
            datasets.append(dataset)

        elif dataset_name.lower() in ["egobody_semiold"]:
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")

            # get collect_fn
            #collate_fn = get_collate_ego(dataset_name)
            # get dataset module
            #! mean = np.load(pjoin(eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT"), "Mean.npy"))
            #! std = np.load(pjoin(eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT"), "Std.npy"))
            
            motion_dir = pjoin(eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT"), "our_process_smpl")
            dr_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            dt_file = 'annotation_egocentric_smpl_npz/egocapture_train_smpl.npz'
            dataset = EgoBodyDataModule(
                #dataset_file=dt_file,
                #data_root=dr_root,
                datapath=eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT"),
                cfg=cfg,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                #collate_fn=collate_fn,
                num_seq_max=cfg.DATASET.SAMPLER.MAX_SQE
                if not cfg.DEBUG else 100,
                mean=None,
                std=None,
                motion_dir=motion_dir,
                )
            datasets.append(dataset)
        else:
            raise NotImplementedError
    cfg.DATASET.NFEATS = datasets[0].nfeats
    cfg.DATASET.NJOINTS = datasets[0].njoints
    return datasets
