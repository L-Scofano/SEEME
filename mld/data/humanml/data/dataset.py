import codecs as cs
import os
import random
from os.path import join as pjoin
from os.path import basename

import numpy as np
import spacy
import torch
from rich.progress import track
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from scipy.spatial.transform import Rotation as R

from typing import Dict
from yacs.config import CfgNode
from os.path import basename
import pickle as pkl
import smplx
import pandas as pd

from EgoHMR.utils import konia_transform
from EgoHMR.dataloaders.augmentation import generate_image_patch, convert_cvimg_to_tensor

from ..utils.get_opt import get_opt
from ..utils.word_vectorizer import WordVectorizer

from .utils_egobody.other_utils import *
from .utils_egobody.geometry import *
from .utils_egobody.other_utils import *
from .utils_egobody.augmentation import *

from .utils_egobody.geometry import rot_aa


from EgoHMR.utils.konia_transform import rotation_matrix_to_angle_axis as rotmat2aa

import trimesh

from scipy.spatial.transform import Rotation


# import spacy
def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


"""For use of training text-2-motion generative model"""


class Text2MotionDataset(data.Dataset):

    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        min_motion_len = 40 if self.opt.dataset_name == "t2m" else 24

        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                      20)]
                                if (len(n_motion)) < min_motion_len or (
                                        len(n_motion) >= 200):
                                    continue
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                    "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                      to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4:4 + (joints_num - 1) * 3] = std[4:4 +
                                                  (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3:4 +
                (joints_num - 1) * 9] = (std[4 + (joints_num - 1) * 3:4 +
                                             (joints_num - 1) * 9] / 1.0)
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9:4 + (joints_num - 1) * 9 +
                joints_num * 3] = (std[4 + (joints_num - 1) * 9:4 +
                                       (joints_num - 1) * 9 + joints_num * 3] /
                                   1.0)
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = (
                std[4 +
                    (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias)

            assert 4 + (joints_num -
                        1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, "mean.npy"), mean)
            np.save(pjoin(opt.meta_dir, "std.npy"), std)

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.opt.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data[
            "text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if self.opt.is_train:
            if m_length != self.max_length:
                # print("Motion original length:%d_%d"%(m_length, len(motion)))
                if self.opt.unit_length < 10:
                    coin2 = np.random.choice(["single", "single", "double"])
                else:
                    coin2 = "single"
                if len_gap == 0 or (len_gap == 1 and coin2 == "double"):
                    m_length = self.max_length
                    idx = random.randint(0, m_length - self.max_length)
                    motion = motion[idx:idx + self.max_length]
                else:
                    if coin2 == "single":
                        n_m_length = self.max_length + self.opt.unit_length * len_gap
                    else:
                        n_m_length = self.max_length + self.opt.unit_length * (
                            len_gap - 1)
                    idx = random.randint(0, m_length - n_m_length)
                    motion = motion[idx:idx + self.max_length]
                    m_length = n_m_length
                # print(len_gap, idx, coin2)
        else:
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(["single", "single", "double"])
            else:
                coin2 = "single"

            if coin2 == "double":
                m_length = (m_length // self.opt.unit_length -
                            1) * self.opt.unit_length
            elif coin2 == "single":
                m_length = (m_length //
                            self.opt.unit_length) * self.opt.unit_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx + m_length]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length


"""For use of training text motion matching model, and evaluations"""


class Text2MotionDatasetV2(data.Dataset):

    def __init__(
        self,
        mean,
        std,
        split_file,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        text_dir,
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs,
    ):
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        # min_motion_len = 40 if dataset_name =='t2m' else 24
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.id_list = id_list

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10

        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading HumanML3D {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)
        count = 0
        bad_count = 0
        new_name_list = []
        length_list = []
        for i, name in enumerator:
            if count > maxdata:
                break
            try:
                motion = np.load(pjoin(motion_dir, name + ".npy"))
                if (len(motion)) < self.min_motion_length or (len(motion) >=
                                                              200):
                    bad_count += 1
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                      20)]
                                if (len(n_motion)
                                    ) < self.min_motion_length or (
                                        (len(n_motion) >= 200)):
                                    continue
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                    "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                # None
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                      to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    # print(count)
                    count += 1
                    # print(name)
            except:
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data[
            "text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        # # padding
        # if m_length < self.max_motion_length:
        #     motion = np.concatenate(
        #         [
        #             motion,
        #             np.zeros((self.max_motion_length - m_length, motion.shape[1])),
        #         ],
        #         axis=0,
        #     )
        # print(word_embeddings.shape, motion.shape, m_length)
        # print(tokens)

        # debug check nan
        if np.any(np.isnan(motion)):
            raise ValueError("nan in motion")

        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            "_".join(tokens),
        )
        # return caption, motion, m_length


"""For use of training baseline"""


class Text2MotionDatasetBaseline(data.Dataset):

    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name == "t2m" else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                      20)]
                                if (len(n_motion)) < min_motion_len or (
                                        len(n_motion) >= 200):
                                    continue
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                    "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                      to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data[
            "text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if m_length != self.max_length:
            # print("Motion original length:%d_%d"%(m_length, len(motion)))
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(["single", "single", "double"])
            else:
                coin2 = "single"
            if len_gap == 0 or (len_gap == 1 and coin2 == "double"):
                m_length = self.max_length
                s_idx = random.randint(0, m_length - self.max_length)
            else:
                if coin2 == "single":
                    n_m_length = self.max_length + self.opt.unit_length * len_gap
                else:
                    n_m_length = self.max_length + self.opt.unit_length * (
                        len_gap - 1)
                s_idx = random.randint(0, m_length - n_m_length)
                m_length = n_m_length
        else:
            s_idx = 0

        src_motion = motion[s_idx:s_idx + m_length]
        tgt_motion = motion[s_idx:s_idx + self.max_length]
        "Z Normalization"
        src_motion = (src_motion - self.mean) / self.std
        tgt_motion = (tgt_motion - self.mean) / self.std

        # padding
        if m_length < self.max_motion_length:
            src_motion = np.concatenate(
                [
                    src_motion,
                    np.zeros(
                        (self.max_motion_length - m_length, motion.shape[1])),
                ],
                axis=0,
            )
        # print(m_length, src_motion.shape, tgt_motion.shape)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, caption, sent_len, src_motion, tgt_motion, m_length


class MotionDatasetV2(data.Dataset):

    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except:
                # Some motion may not exist in KIT dataset
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4:4 + (joints_num - 1) * 3] = std[4:4 +
                                                  (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3:4 +
                (joints_num - 1) * 9] = (std[4 + (joints_num - 1) * 3:4 +
                                             (joints_num - 1) * 9] / 1.0)
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9:4 + (joints_num - 1) * 9 +
                joints_num * 3] = (std[4 + (joints_num - 1) * 9:4 +
                                       (joints_num - 1) * 9 + joints_num * 3] /
                                   1.0)
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = (
                std[4 +
                    (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias)

            assert 4 + (joints_num -
                        1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, "mean.npy"), mean)
            np.save(pjoin(opt.meta_dir, "std.npy"), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(
            len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion


class RawTextDataset(data.Dataset):

    def __init__(self, opt, mean, std, text_file, w_vectorizer):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.nlp = spacy.load("en_core_web_sm")

        with cs.open(text_file) as f:
            for line in f.readlines():
                word_list, pos_list = self.process_text(line.strip())
                tokens = [
                    "%s/%s" % (word_list[i], pos_list[i])
                    for i in range(len(word_list))
                ]
                self.data_dict.append({
                    "caption": line.strip(),
                    "tokens": tokens
                })

        self.w_vectorizer = w_vectorizer
        print("Total number of descriptions {}".format(len(self.data_dict)))

    def process_text(self, sentence):
        sentence = sentence.replace("-", "")
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == "NOUN"
                    or token.pos_ == "VERB") and (word != "left"):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[item]
        caption, tokens = data["caption"], data["tokens"]

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len


class TextOnlyDataset(data.Dataset):

    def __init__(self, opt, mean, std, split_file, text_dir, **kwargs):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.max_length = 20
        self.pointer = 0
        self.fixed_length = 120

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                    "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {"text": [text_dict]}
                                new_name_list.append(new_name)
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                      to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {"text": text_data}
                    new_name_list.append(name)
            except:
                pass

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        text_list = data["text"]

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]
        return None, None, caption, None, np.array([0
                                                    ]), self.fixed_length, None
        # fixed_length can be set from outside before sampling


# A wrapper class for t2m original dataset for MDM purposes
class HumanML3D(data.Dataset):

    def __init__(self,
                 mode,
                 datapath="./dataset/humanml_opt.txt",
                 split="train",
                 **kwargs):
        self.mode = mode

        self.dataset_name = "t2m"
        self.dataname = "t2m"

        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = f"."
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = (
            None  # torch.device('cuda:4') # This param is not in use in this context
        )
        opt = get_opt(dataset_opt_path, device)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        self.opt = opt
        print("Loading dataset %s ..." % opt.dataset_name)

        if mode == "gt":
            # used by T2M models (including evaluators)
            self.mean = np.load(pjoin(opt.meta_dir, "mean.npy"))
            self.std = np.load(pjoin(opt.meta_dir, "std.npy"))
        elif mode in ["train", "eval", "text_only"]:
            # used by our models
            self.mean = np.load(pjoin(opt.data_root, "Mean.npy"))
            self.std = np.load(pjoin(opt.data_root, "Std.npy"))

        if mode == "eval":
            # used by T2M models (including evaluators)
            # this is to translate their norms to ours
            self.mean_for_eval = np.load(pjoin(opt.meta_dir, "mean.npy"))
            self.std_for_eval = np.load(pjoin(opt.meta_dir, "std.npy"))

        self.split_file = pjoin(opt.data_root, f"{split}.txt")
        if mode == "text_only":
            self.t2m_dataset = TextOnlyDataset(self.opt, self.mean, self.std,
                                               self.split_file)
        else:
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, "glove"),
                                               "our_vab")
            self.t2m_dataset = Text2MotionDatasetV2(self.opt, self.mean,
                                                    self.std, self.split_file,
                                                    self.w_vectorizer)
            self.num_actions = 1  # dummy placeholder

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()


# A wrapper class for t2m original dataset for MDM purposes
class KIT(HumanML3D):

    def __init__(self,
                 mode,
                 datapath="./dataset/kit_opt.txt",
                 split="train",
                 **kwargs):
        super(KIT, self).__init__(mode, datapath, split, **kwargs)


class EgoBodyData(data.Dataset):

    def __init__(
            self,
            mean, 
            std,
            split_file,
            motion_dir,
            #max_motion_length,
            #min_motion_length,
            tiny=False,
            debug=False,
            progress_bar=True,
            **kwargs,
    ):
        
        #self.max_motion_length = max_motion_length
        #self.min_motion_length = min_motion_length

        split = split_file.split("/")[-1].split(".")[0]
        motion_dir = pjoin(motion_dir, split)

       

        self.mean = mean
        self.std = std
        data_dict = {}
        #id_list = []
        #with cs.open(split_file, "r") as f:
        #    for line in f.readlines():
        #        id_list.append(line.strip())
        #self.id_list = id_list

        id_list = os.listdir(motion_dir)

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10


        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading EgoBody {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)

        count = 0
        bad_count = 0
        new_name_list = []
        length_list = []
        for i, name in enumerator:
            if count > maxdata:
                break
            try:
                motion = np.load(pjoin(motion_dir, name))
                flag = False
                data_dict[name] = {
                    "motion": motion
                }
                new_name_list.append(name)
                length_list.append(len(motion))
                count += 1
            except:
                pass
        
        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list

    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, item):
        idx = item
        data = self.data_dict[self.name_list[idx]]
        motion = data["motion"]

        # Z normalization
        motion = (motion - self.mean) / self.std

        #m_length = len(motion)
        return motion



    
class EgoBodyData3(data.Dataset):

    def __init__(
            self,
            mean, 
            std,
            split_file,
            motion_dir,
            condition=None,
            interactee_pred=None,
            pred_global_orient=False,
            pred_betas=False,
            transl_egoego=False,
            global_orient_egoego=False,
            predict_transl=False,
            motion_length=60,
            data_type='angle',
            droid_slam_cut=False,
            pose_estimation_task=False,
            #max_motion_length,
            #min_motion_length,
            tiny=False,
            debug=False,
            progress_bar=True,
            **kwargs,
    ):
        
        #self.max_motion_length = max_motion_length
        #self.min_motion_length = min_motion_length
      
        if data_type == 'angle':
            self.numdims = 72
            self.go_dims = 3
            self.mean = np.load('./datasets/EgoBody/our_process_smpl_split_NEW/mean.npy') #mean
            self.std = np.load('./datasets/EgoBody/our_process_smpl_split_NEW/std.npy') #std
        elif data_type == 'rot6d':
            self.numdims = 144
            self.go_dims = 6
            self.mean = np.load('./datasets/EgoBody/our_process_smpl_split_NEW/mean_rot6d.npy') #mean
            self.std = np.load('./datasets/EgoBody/our_process_smpl_split_NEW/std_rot6d.npy') #std


        split = split_file.split("/")[-1].split(".")[0]
        self.split = split
        self.droid_slam_cut = droid_slam_cut
        if droid_slam_cut:
            motion_dir = pjoin(motion_dir, split + '_droidslam_8') if split=='test' else pjoin(motion_dir, split)
        else:
            motion_dir = pjoin(motion_dir, split) # './datasets/EgoBody/our_process_smpl_split_NEW/val'

        self.condition = condition
        self.interactee_pred = interactee_pred
        self.pred_global_orient = pred_global_orient
        self.pred_betas = pred_betas

        self.global_orient_egoego = global_orient_egoego
        self.transl_egoego = transl_egoego

        self.predict_transl = predict_transl

        self.pose_estimation_task = pose_estimation_task

        self.motion_length = motion_length

        self.data_type = data_type

        #self.mean = mean
        #self.std = std
        
        data_dict = {}
        #id_list = []
        #with cs.open(split_file, "r") as f:
        #    for line in f.readlines():
        #        id_list.append(line.strip())
        #self.id_list = id_list

        id_list = os.listdir(motion_dir)
        #if self.droid_slam_cut:
        #    id_list = id_list[:64]

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10

        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading EgoBody {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)

        count = 0
        bad_count = 0
        new_name_list = []
        length_list = []
        for i, name in enumerator:
            if count > maxdata:
                break
            try:
                motion = np.load(pjoin(motion_dir, name), allow_pickle=True).item()
                data_dict[name] = {
                    "video": motion['video'],
                    "recording_utils": motion['recording_utils'],
                    "interactee": motion['interactee'],
                    'wearer': motion['wearer'],
                }
                
                new_name_list.append(name)
                length_list.append(len(motion))
                count += 1
            except:
                pass
        
        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = 69 #motion.shape[1]
        self.name_list = name_list

        if 'scene' in condition:
            self.add_trans = np.array([[1.0, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            map_path = './datasets/EgoBody/Egohmr_scene_preprocess_s1_release/map_dict_{}.pkl'.format(split.split('.')[0])
            with open(map_path, 'rb') as f:
                self.scene_map_dict = pkl.load(f)
            pcd_path = './datasets/EgoBody/Egohmr_scene_preprocess_s1_release/pcd_verts_dict_{}.pkl'.format(split.split('.')[0])
            with open(pcd_path, 'rb') as f:
                self.scene_verts_dict = pkl.load(f)
            with open(os.path.join('./datasets/EgoBody/', 'transf_matrices_all_seqs.pkl'), 'rb') as fp:
                self.transf_matrices = pkl.load(fp)
        
        if self.interactee_pred:
            self.interactee_pred_dict = {}
            with open(os.path.join('./datasets/EgoBody/results_egohmr/results_interactee_test.pkl'), 'rb') as fp:
                self.interactee_pred_dict = pkl.load(fp)

        if self.global_orient_egoego or self.transl_egoego:
            file_ = './datasets/EgoBody/trans_and_rot_pred/data.pkl'
            with open(file_, 'rb') as fp:
                self.pred_egoego = pkl.load(fp)


    def __len__(self):
        return len(self.name_list)
    

    
    def get_transf_matrices_per_frame(self, timestamp, seq_name):
        transf_mtx_seq = self.transf_matrices[seq_name]
        kinect2holo = transf_mtx_seq['trans_kinect2holo'].astype(np.float32)  # [4,4], one matrix for all frames in the sequence
        holo2pv_dict = transf_mtx_seq['trans_world2pv']  # a dict, # frames items, each frame is a 4x4 matrix
        #timestamp = basename(img_name).split('_')[0]
        holo2pv = holo2pv_dict[str(timestamp)].astype(np.float32)
        return kinect2holo, holo2pv
    
    def __getitem__(self, item):
        idx = item
        data = self.data_dict[self.name_list[idx]]
        #motion = data["motion"]

        

        video = data['video']
        recording_utils = data['recording_utils']
        interactee = data['interactee']
        wearer = data['wearer']


        if self.pose_estimation_task:
            interactee_gt_pose_estimation = data['interactee']

        list_imgname = []
        for imgname in recording_utils['original_imgname']:
            list_imgname.append(str(imgname))
        
        #img_name_dict = torch.tensor(list_imgname)
        #list_imgname = recording_utils['original_imgname']
        

        if 'scene' in self.condition:
            original_imagename = recording_utils['original_imgname'][0]
            seq_name = original_imagename.split('/')[1]
            timestamp = original_imagename.split('/')[4].split('_')[0]
            
            transf_kinect2holo, transf_holo2pv = self.get_transf_matrices_per_frame(timestamp, seq_name)
            pcd_trans_kinect2pv = np.matmul(transf_holo2pv, transf_kinect2holo)
            pcd_trans_kinect2pv = np.matmul(self.add_trans, pcd_trans_kinect2pv)

            # WHOLE-SCENE
            scene_pcd_verts = self.scene_verts_dict[self.scene_map_dict[original_imagename]]  # [20000, 3], in kinect main coord
            scene_pcd_verts = points_coord_trans(scene_pcd_verts, pcd_trans_kinect2pv)
            # to tensor
            scene_pcd_verts = torch.tensor(scene_pcd_verts, dtype=torch.float32) # [20000, 3]       
            
        # Remove key ['original_imgname'] from recording_utils
        #recording_utils.pop('original_imgname', None)

        utils_ = {'video': video, 'recording_utils': recording_utils}

    
        fx = torch.tensor(np.array(recording_utils['fx'])).reshape(-1,1)
        cx = torch.tensor(np.array(recording_utils['cx'])).reshape(-1,1)
        cy = torch.tensor(np.array(recording_utils['cy'])).reshape(-1,1)
        center = torch.tensor(np.array(recording_utils['center'])).reshape(-1,2)
        scale = torch.tensor(np.array(recording_utils['scale'])).reshape(-1,1)
        #utils = torch.cat([fx, cx, cy, center, scale], dim=1)
        
        item_length = len(video)

        if self.interactee_pred:
            images_in_batch = recording_utils['original_imgname']
            full_go = []
            full_bp = []
            full_betas = []
            
            #print(list_imgname)
            for image in images_in_batch:
                interactee_gt = self.interactee_pred_dict[image]
                global_orient_gt = interactee_gt['smpl_parameters']['global_orient']
                body_pose_gt = interactee_gt['smpl_parameters']['body_pose']
                betas_gt = interactee_gt['smpl_parameters']['betas']
                full_go.append(global_orient_gt)
                full_bp.append(body_pose_gt)
                full_betas.append(betas_gt)

        

            smpl_params_interactee = {'global_orient': np.array(full_go[:item_length]).reshape(-1,1,3),
                          'transl': np.array(interactee['transl']), # !!! NOT FROM EgoHMR !!!
                          'body_pose': np.array(full_bp[:item_length]).reshape(-1,1,69),#.reshape(-1,23,3),
                          'betas': np.array(full_betas[:item_length]).reshape(-1,1,10),
                         }
                                      
            
        else:
            smpl_params_interactee = {'global_orient': np.array(interactee['global_orient']),
                       'transl': np.array(interactee['transl']),
                       'body_pose': np.array(interactee['body_pose']),
                       'betas': np.array(interactee['betas']),
                      }

        if self.pose_estimation_task:
            smpl_params_interactee_pe_gt = {'global_orient': np.array(interactee_gt_pose_estimation['global_orient']),
                       'transl': np.array(interactee_gt_pose_estimation['transl']),
                       'body_pose': np.array(interactee_gt_pose_estimation['body_pose']),
                       'betas': np.array(interactee_gt_pose_estimation['betas']),
                      }


        if self.global_orient_egoego or self.transl_egoego:
            pred_egoego_transl = []
            pred_egoego_go = []
            for i, img_path in enumerate(recording_utils['original_imgname']):
                try:
                    trns = self.pred_egoego[img_path]['transl']
                    gor = self.pred_egoego[img_path]['global_orient']
                except:
                    trns = pred_egoego_transl[-1] if len(pred_egoego_transl) > 0 else [0., 0., 0.]
                    gor = pred_egoego_go[-1] if len(pred_egoego_go) > 0 else [[0.,0.,0.]*3]

                pred_egoego_transl.append(trns)
                pred_egoego_go.append(gor)
            pred_egoego_transl = torch.tensor(np.array(pred_egoego_transl))

            pred_egoego_go = torch.tensor(np.array(pred_egoego_go))
            pred_egoego_global_orient = rotmat2aa(pred_egoego_go.reshape(-1, 3, 3))


            

        smpl_params_wearer = {'global_orient': np.array(wearer['global_orient']),
                              'transl': np.array(wearer['transl']),
                                'body_pose': np.array(wearer['body_pose']),
                                'betas': np.array(wearer['betas']),
                                }
        
        if self.data_type == 'rot6d':
            out_rot_wear_bp = aa_to_rotmat(torch.tensor(smpl_params_wearer['body_pose'].reshape(-1,23,3)).reshape(-1, 3)).view(-1, 23, 3, 3)
            out_rot_wear_go = aa_to_rotmat(torch.tensor(smpl_params_wearer['global_orient'].reshape(-1,1,3)).reshape(-1, 3)).view(-1, 1, 3, 3)

            out_rot_int_bp = aa_to_rotmat(torch.tensor(smpl_params_interactee['body_pose'].reshape(-1,23,3)).reshape(-1, 3)).view(-1, 23, 3, 3)
            out_rot_int_go = aa_to_rotmat(torch.tensor(smpl_params_interactee['global_orient'].reshape(-1,1,3)).reshape(-1, 3)).view(-1, 1, 3, 3)

            out_6d_wear_bp = rotmat_to_rot6d(out_rot_wear_bp.reshape(-1, 3, 3)).reshape(-1, 1, 23*6)
            out_6d_wear_go = rotmat_to_rot6d(out_rot_wear_go.reshape(-1, 3, 3)).reshape(-1, 1, 6)

            out_6d_int_bp = rotmat_to_rot6d(out_rot_int_bp.reshape(-1, 3, 3)).reshape(-1, 1, 23*6)
            out_6d_int_go = rotmat_to_rot6d(out_rot_int_go.reshape(-1, 3, 3)).reshape(-1, 1, 6)

            smpl_params_interactee['body_pose'] = out_6d_int_bp.numpy()
            smpl_params_interactee['global_orient'] = out_6d_int_go.numpy()
            smpl_params_wearer['body_pose'] = out_6d_wear_bp.numpy()
            smpl_params_wearer['global_orient'] = out_6d_wear_go.numpy()


            '''
            wear_bp = rotmat_to_rot6d(aa_to_rotmat(torch.tensor(smpl_params_wearer['body_pose']).reshape(-1, 3)).reshape(-1, 23, 3, 3)).reshape(-1, 1, 23*6)
            wear_go = rotmat_to_rot6d(aa_to_rotmat(torch.tensor(smpl_params_wearer['global_orient']).reshape(-1, 3)).reshape(-1, 1, 3, 3)).reshape(-1, 1, 6)

            interactee_bp = rotmat_to_rot6d(aa_to_rotmat(torch.tensor(smpl_params_interactee['body_pose']).reshape(-1, 3)).reshape(-1, 23, 3, 3)).reshape(-1, 1, 23*6)
            interactee_go = rotmat_to_rot6d(aa_to_rotmat(torch.tensor(smpl_params_interactee['global_orient']).reshape(-1, 3)).reshape(-1, 1, 3, 3)).reshape(-1, 1, 6)
            
            smpl_params_interactee['body_pose'] = interactee_bp.numpy()
            smpl_params_interactee['global_orient'] = interactee_go.numpy()
            smpl_params_wearer['body_pose'] = wear_bp.numpy()
            smpl_params_wearer['global_orient'] = wear_go.numpy()'''
        
        elif self.data_type == 'angle':
            # Nedd to # apply the global rotation to the global orientation
            global_orient_int = smpl_params_interactee['global_orient']
            global_orient_wear = smpl_params_wearer['global_orient']


            

            #for i in range(global_orient_int.shape[0]):
            #    global_orient_int[i] = rot_aa(global_orient_int[i].reshape(-1, 3), rot=0.).reshape(-1, 1, 3)
            #    global_orient_wear[i] = rot_aa(global_orient_wear[i].reshape(-1, 3), rot=0.).reshape(-1, 1, 3)
            

            smpl_params_interactee['global_orient'] = global_orient_int
            smpl_params_wearer['global_orient'] = global_orient_wear

            if self.pose_estimation_task:
                global_orient_int_pe_gt = smpl_params_interactee_pe_gt['global_orient']
                smpl_params_interactee_pe_gt['global_orient'] = global_orient_int_pe_gt
           


        has_smpl_params = {'global_orient': True,
                           'transl': True,
                           'body_pose': True,
                           'betas': True
                           }
        
        smpl_params_is_axis_angle = {'global_orient': True,
                                     'transl': False,
                                     'body_pose': True,
                                     'betas': False
                                    }


        do_like_egoego=False
        normalize_altogether=False
        if normalize_altogether:
            mean = self.mean[0, :self.numdims]
            std = self.std[0, :self.numdims]
            full_pose_aa_interactee = np.concatenate([smpl_params_interactee['global_orient'], 
                                                    smpl_params_interactee['body_pose']], axis=-1)
            full_pose_aa_interactee = (full_pose_aa_interactee.reshape(self.motion_length, -1) - mean) / std
            

            full_pose_aa_wearer = np.concatenate([smpl_params_wearer['global_orient'],
                                                    smpl_params_wearer['body_pose']], axis=-1)
            full_pose_aa_wearer = (full_pose_aa_wearer.reshape(self.motion_length, -1) - mean) / std
            
            interactee = torch.tensor(full_pose_aa_interactee, dtype=torch.float32).unsqueeze(1)
            wearer = torch.tensor(full_pose_aa_wearer, dtype=torch.float32).unsqueeze(1)

            motion = torch.cat([wearer, interactee], dim=1)
        
        else:
            
            full_pose_aa_interactee = smpl_params_interactee['body_pose']
            full_pose_aa_wearer = smpl_params_wearer['body_pose']

            if self.pose_estimation_task:
                full_pose_aa_interactee_pe_gt = smpl_params_interactee_pe_gt['body_pose']



            mean = self.mean[0, self.go_dims:self.numdims]
            std = self.std[0, self.go_dims:self.numdims]

            
            if item_length!=self.motion_length:
                length_to_pad = self.motion_length - item_length
                full_pose_aa_interactee = np.concatenate([full_pose_aa_interactee, np.zeros((length_to_pad, 1, 69))], axis=0)
                full_pose_aa_wearer = np.concatenate([full_pose_aa_wearer, np.zeros((length_to_pad, 1, 69))], axis=0)
                if self.pose_estimation_task:
                    full_pose_aa_interactee_pe_gt = np.concatenate([full_pose_aa_interactee_pe_gt, np.zeros((length_to_pad, 1, 69))], axis=0)

            full_pose_aa_interactee = (full_pose_aa_interactee.reshape(self.motion_length, -1) - mean) / std
            full_pose_aa_wearer = (full_pose_aa_wearer.reshape(self.motion_length, -1) - mean) / std
            if self.pose_estimation_task:
                full_pose_aa_interactee_pe_gt = (full_pose_aa_interactee_pe_gt.reshape(self.motion_length, -1) - mean) / std
            
            interactee = torch.tensor(full_pose_aa_interactee, dtype=torch.float32).unsqueeze(1)
            wearer = torch.tensor(full_pose_aa_wearer, dtype=torch.float32).unsqueeze(1)

            if self.pose_estimation_task:
                interactee_pe_gt = torch.tensor(full_pose_aa_interactee_pe_gt, dtype=torch.float32).unsqueeze(1)

            motion = torch.cat([wearer, interactee], dim=1)

            #! NOT NORMALIZED
            interactee_go = torch.tensor(smpl_params_interactee['global_orient'], dtype=torch.float32)
            wearer_go = torch.tensor(smpl_params_wearer['global_orient'], dtype=torch.float32)

            if self.pose_estimation_task:
                interactee_go_pe_gt = torch.tensor(smpl_params_interactee_pe_gt['global_orient'], dtype=torch.float32)

            if item_length!=self.motion_length:
                interactee_go = torch.cat([interactee_go, torch.zeros((length_to_pad, 1, 3))], dim=0)
                wearer_go = torch.cat([wearer_go, torch.zeros((length_to_pad, 1, 3))], dim=0)

                if self.pose_estimation_task:
                    interactee_go_pe_gt = torch.cat([interactee_go_pe_gt, torch.zeros((length_to_pad, 1, 3))], dim=0)

            
            mean = self.mean[0, :self.go_dims]
            std = self.std[0, :self.go_dims]
            interactee_go = (interactee_go - mean) / std
            wearer_go = (wearer_go - mean) / std

            if self.pose_estimation_task:
                interactee_go_pe_gt = (interactee_go_pe_gt - mean) / std

            
            global_orient = torch.cat([wearer_go, interactee_go], dim=1)
            
            motion = torch.cat([global_orient, motion], dim=-1)
            if self.pose_estimation_task:
                motion_interacte_pe_gt = torch.cat([interactee_go_pe_gt, interactee_pe_gt], dim=-1)

        utils = torch.cat([fx, cx, cy, center, scale], dim=1)
        if item_length!=self.motion_length:
            utils = torch.cat([utils, torch.zeros((length_to_pad, 6))], dim=0)

                
        #betas = torch.cat([torch.tensor(smpl_params_wearer['betas'], dtype=torch.float32).unsqueeze(0),
        #                    torch.tensor(smpl_params_interactee['betas'], dtype=torch.float32).unsqueeze(0)], dim=0)

        interactee_transl = torch.tensor(smpl_params_interactee['transl'], dtype=torch.float32)
        wearer_transl = torch.tensor(smpl_params_wearer['transl'], dtype=torch.float32)

        if self.pose_estimation_task:
            interactee_transl_pe_gt = torch.tensor(smpl_params_interactee_pe_gt['transl'], dtype=torch.float32)

        if item_length!=self.motion_length:
            interactee_transl = torch.cat([interactee_transl, torch.zeros((length_to_pad, 1, 3))], dim=0)
            wearer_transl = torch.cat([wearer_transl, torch.zeros((length_to_pad, 1, 3))], dim=0)
            if self.pose_estimation_task:
                interactee_transl_pe_gt = torch.cat([interactee_transl_pe_gt, torch.zeros((length_to_pad, 1, 3))], dim=0)


        if self.predict_transl and self.data_type == 'angle':
            interactee_transl = (interactee_transl - self.mean[0, self.numdims:self.numdims+3]) / self.std[0, self.numdims:self.numdims+3]
            wearer_transl = (wearer_transl - self.mean[0, self.numdims:self.numdims+3]) / self.std[0, self.numdims:self.numdims+3]
            if self.pose_estimation_task:
                interactee_transl_pe_gt = (interactee_transl_pe_gt - self.mean[0, self.numdims:self.numdims+3]) / self.std[0, self.numdims:self.numdims+3]

        transl = torch.cat([wearer_transl, interactee_transl], dim=1).permute(1, 0, 2)
        if self.pose_estimation_task:
            interactee_transl_pe_gt = interactee_transl_pe_gt.permute(1, 0, 2)

        interactee_beta = torch.tensor(smpl_params_interactee['betas'], dtype=torch.float32)
        wearer_beta = torch.tensor(smpl_params_wearer['betas'], dtype=torch.float32)
        if self.pose_estimation_task:
            interactee_beta_pe_gt = torch.tensor(smpl_params_interactee_pe_gt['betas'], dtype=torch.float32)

        if item_length!=self.motion_length:
            interactee_beta = torch.cat([interactee_beta, torch.zeros((length_to_pad, 1, 10))], dim=0)
            wearer_beta = torch.cat([wearer_beta, torch.zeros((length_to_pad, 1, 10))], dim=0)
            if self.pose_estimation_task:
                interactee_beta_pe_gt = torch.cat([interactee_beta_pe_gt, torch.zeros((length_to_pad, 1, 10))], dim=0).permute(1, 0, 2)

        beta = torch.cat([wearer_beta, interactee_beta], dim=1).permute(1, 0, 2)

        length = torch.tensor([item_length], dtype=torch.int32)
        
        
        if 'image' in self.condition or self.pred_betas:
            images = []
            number_of_images = len(recording_utils['original_imgname'])
            image_idx = np.random.randint(0, number_of_images)
            img_path = os.path.join('./datasets/EgoBody/', recording_utils['original_imgname'][image_idx])
            cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            img_height, img_width, img_channels = cvimg.shape
            width, height = 224, 224
            
            bbox_size = recording_utils['scale'][image_idx]*200
            center_x = recording_utils['center'][image_idx][0] + bbox_size
            center_y = recording_utils['center'][image_idx][1] + bbox_size

            img_patch_cv, trans_crop = generate_image_patch(cvimg,
                                            center_x, center_y,
                                            bbox_size, bbox_size,
                                            patch_width=width, patch_height=height,
                                            do_flip=False, scale=1.0, rot=0.)
            image = img_patch_cv.copy()

            
            image = image[:, :, ::-1]  # [224, 224, 3] BGR-->RGB

            img_patch_cv = image.copy()
            img_patch = convert_cvimg_to_tensor(image)

            # apply RGB normalization
            color_scale = [1.0, 1.0, 1.0]
            mean_col, std_col = 255.*np.array([0.485, 0.456, 0.406]), 255. *np.array([0.229, 0.224, 0.225])
            for n_c in range(img_channels):
                img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)
                if mean is not None and std is not None:
                    img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean_col[n_c]) / std_col[n_c]
            images = torch.tensor(img_patch, dtype=torch.float32)
            '''
            for i, img_path in enumerate(recording_utils['original_imgname']):
                img_path = os.path.join('./datasets/EgoBody/', img_path)
                cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                img_height, img_width, img_channels = cvimg.shape
                width, height = 224, 224
                
                bbox_size = recording_utils['scale'][i]*200
                center_x = recording_utils['center'][i][0] + bbox_size
                center_y = recording_utils['center'][i][1] + bbox_size

                img_patch_cv, trans_crop = generate_image_patch(cvimg,
                                                center_x, center_y,
                                                bbox_size, bbox_size,
                                                patch_width=width, patch_height=height,
                                                do_flip=False, scale=1.0, rot=0.)
                image = img_patch_cv.copy()

                
                image = image[:, :, ::-1]  # [224, 224, 3] BGR-->RGB

                img_patch_cv = image.copy()
                img_patch = convert_cvimg_to_tensor(image)

                # apply RGB normalization
                color_scale = [1.0, 1.0, 1.0]
                mean_col, std_col = 255.*np.array([0.485, 0.456, 0.406]), 255. *np.array([0.229, 0.224, 0.225])
                for n_c in range(img_channels):
                    img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)
                    if mean is not None and std is not None:
                        img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean_col[n_c]) / std_col[n_c]
                images.append(img_patch)
            images = np.stack(images, axis=0)
            images = torch.tensor(images, dtype=torch.float32)

            nuber_of_images = images.shape[0]
            # Sample an image
            image_idx = np.random.randint(0, nuber_of_images)
            images = images[image_idx]'''


        
                

        
        if 'scene' in self.condition and 'image' not in self.condition:
            #if self.pred_betas:
            #    return motion, transl, beta, utils, scene_pcd_verts, images
            #else:
            #    return motion, transl, beta, utils, scene_pcd_verts
            
            if self.transl_egoego:
                return motion, transl, beta, utils, scene_pcd_verts, pred_egoego_transl, pred_egoego_global_orient, length
            else:
                if self.pose_estimation_task:
                    return motion, transl, beta, utils, scene_pcd_verts, length, motion_interacte_pe_gt, interactee_transl_pe_gt, interactee_beta_pe_gt 
                else:
                    return motion, transl, beta, utils, scene_pcd_verts, length, list_imgname

        elif 'scene' in self.condition and 'image' in self.condition:
            
            return motion, transl, beta, utils, scene_pcd_verts, images, length
        elif 'scene' not in self.condition and 'image' in self.condition:
            return motion, transl, beta, utils, images, length
        else:
            return motion, transl, beta, utils, length
        
class GimoData(data.Dataset):

    def __init__(
            self,
            mean, 
            std,
            split_file,
            motion_dir,
            condition=None,
            interactee_pred=None,
            pred_global_orient=False,
            pred_betas=False,
            transl_egoego=False,
            global_orient_egoego=False,
            predict_transl=False,
            motion_length=60,
            data_type='angle',
            droid_slam_cut=False,
            pose_estimation_task=False,
            #max_motion_length,
            #min_motion_length,
            tiny=False,
            debug=False,
            progress_bar=True,
            **kwargs,
    ):
        
        #self.max_motion_length = max_motion_length
        #self.min_motion_length = min_motion_length
      
        if data_type == 'angle':
            self.numdims = 66 #66 # 21*3+3+3
            self.go_dims = 3
            self.mean = np.load('./datasets/GIMO/processed/mean.npy') #mean
            self.std = np.load('./datasets/GIMO/processed/std.npy') #std
        elif data_type == 'rot6d':
            self.numdims = 132
            self.go_dims = 6
            self.mean = np.load('./datasets/GIMO/processed/mean_rot6d.npy') #mean
            self.std = np.load('./datasets/GIMO/processed/std_rot6d.npy') #std


        split = split_file.split("/")[-1].split(".")[0]
        self.split = split
        # !Gimo does not contain any val split
        if split == 'val':
            split = 'test'
        self.droid_slam_cut = droid_slam_cut
        if droid_slam_cut:
            motion_dir = pjoin(motion_dir, split + '_droidslam_8') if split=='test' else pjoin(motion_dir, split)
        else:
            motion_dir = pjoin(motion_dir, split) # './datasets/EgoBody/our_process_smpl_split_NEW/val'

        self.condition = condition
        self.interactee_pred = interactee_pred
        self.pred_global_orient = pred_global_orient
        self.pred_betas = pred_betas

        self.global_orient_egoego = global_orient_egoego
        self.transl_egoego = transl_egoego

        self.predict_transl = predict_transl

        self.pose_estimation_task = pose_estimation_task

        self.motion_length = motion_length

        self.data_type = data_type

        #self.mean = mean
        #self.std = std
        
        data_dict = {}
        #id_list = []
        #with cs.open(split_file, "r") as f:
        #    for line in f.readlines():
        #        id_list.append(line.strip())
        #self.id_list = id_list

        id_list = os.listdir(motion_dir)
        #if self.droid_slam_cut:
        #    id_list = id_list[:64]

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10

        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading Gimo {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)

        count = 0
        bad_count = 0
        new_name_list = []
        length_list = []
        for i, name in enumerator:
            if count > maxdata:
                break
            try:
                motion = np.load(pjoin(motion_dir, name), allow_pickle=True).item()
                data_dict[name] = {
                    "video": motion['video'],
                    "recording_utils": motion['recording_utils'],
                    "interactee": motion['interactee'],
                    'wearer': motion['wearer'],
                }
                
                new_name_list.append(name)
                length_list.append(len(motion))
                count += 1
            except:
                pass
        
        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = 63 # 21*3 (check?)
        self.name_list = name_list

        # if 'scene' in condition:
        #     self.add_trans = np.array([[1.0, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        #     map_path = './datasets/EgoBody/Egohmr_scene_preprocess_s1_release/map_dict_{}.pkl'.format(split.split('.')[0])
        #     with open(map_path, 'rb') as f:
        #         self.scene_map_dict = pkl.load(f)
        #     pcd_path = './datasets/EgoBody/Egohmr_scene_preprocess_s1_release/pcd_verts_dict_{}.pkl'.format(split.split('.')[0])
        #     with open(pcd_path, 'rb') as f:
        #         self.scene_verts_dict = pkl.load(f)
        #     with open(os.path.join('./datasets/EgoBody/', 'transf_matrices_all_seqs.pkl'), 'rb') as fp:
        #         self.transf_matrices = pkl.load(fp)
        
        # if self.interactee_pred:
        #     self.interactee_pred_dict = {}
        #     with open(os.path.join('./datasets/EgoBody/results_egohmr/results_interactee_test.pkl'), 'rb') as fp:
        #         self.interactee_pred_dict = pkl.load(fp)

        # if self.global_orient_egoego or self.transl_egoego:
        #     file_ = './datasets/EgoBody/trans_and_rot_pred/data.pkl'
        #     with open(file_, 'rb') as fp:
        #         self.pred_egoego = pkl.load(fp)


    def __len__(self):
        return len(self.name_list)
    

    
    def get_transf_matrices_per_frame(self, timestamp, seq_name):
        transf_mtx_seq = self.transf_matrices[seq_name]
        kinect2holo = transf_mtx_seq['trans_kinect2holo'].astype(np.float32)  # [4,4], one matrix for all frames in the sequence
        holo2pv_dict = transf_mtx_seq['trans_world2pv']  # a dict, # frames items, each frame is a 4x4 matrix
        #timestamp = basename(img_name).split('_')[0]
        holo2pv = holo2pv_dict[str(timestamp)].astype(np.float32)
        return kinect2holo, holo2pv
    
    def __getitem__(self, item):
        idx = item
        data = self.data_dict[self.name_list[idx]]
        #motion = data["motion"]

        

        video = data['video']
        recording_utils = data['recording_utils']
        interactee = data['interactee']
        wearer = data['wearer']


        if self.pose_estimation_task:
            interactee_gt_pose_estimation = data['interactee']

        list_imgname = []
        for imgname in recording_utils['original_imgname']:
            list_imgname.append(str(imgname))
        
        #img_name_dict = torch.tensor(list_imgname)
        #list_imgname = recording_utils['original_imgname']
        
        # * Load the scene point cloud
        if 'scene' in self.condition:
            original_imagename = data['video'][0]
            scene = original_imagename.split('/')[-4]
            temp_root_scene = 'datasets/gimo_raw/group/GIMO' # ! This path needs to be added to the config file

            scale = 1.03
            transform_norm = np.loadtxt(os.path.join(temp_root_scene, scene, 'scene_obj', 'transform_norm.txt')).reshape(
            (4, 4))
            transform_norm[:3, 3] /= scale

            random_ori = 0
            random_rotation = Rotation.from_euler('xyz', [0, random_ori, 0], degrees=True).as_matrix()
            
            # transform_path = self.dataset_info['transformation'][i]
            # start_frame = self.dataset_info['start_frame'][i]
            # transform_info = json.load(open(os.path.join(self.dataroot, scene, seq, transform_path), 'r'))
            # scale = transform_info['scale']
            # trans_pose2scene = np.array(transform_info['transformation'])
            # trans_scene2pose = np.linalg.inv(trans_pose2scene)

            scene_ply = trimesh.load_mesh(os.path.join(temp_root_scene, scene, 'scene_obj', 'scene_downsampled.ply'))
            scene_points = scene_ply.vertices
            scene_points = scene_points[np.random.choice(range(len(scene_points)), 20000)] # 20000 sampled points in the Point Cloud
    
            # Rescale the scene point cloud
            scene_points *= 1/scale
            scene_points = (transform_norm[:3, :3] @ scene_points.T + transform_norm[:3, 3:]).T

            if self.split == 'train':
                sigma = 0.01
                scene_points = (random_rotation @ scene_points.T).T
                scene_points += np.random.normal(loc=0, scale=sigma, size=scene_points.shape)

            scene_pcd_verts = torch.from_numpy(scene_points).float()
                



        

        utils_ = {'video': video, 'recording_utils': recording_utils}

    
        fx = torch.tensor(np.array(recording_utils['fx'])).reshape(-1,1)
        cx = torch.tensor(np.array(recording_utils['cx'])).reshape(-1,1)
        cy = torch.tensor(np.array(recording_utils['cy'])).reshape(-1,1)
        center = torch.tensor(np.array(recording_utils['center'])).reshape(-1,2)
        scale = torch.tensor(np.array(recording_utils['scale'])).reshape(-1,1)
        #utils = torch.cat([fx, cx, cy, center, scale], dim=1)
        
        item_length = len(video)

        # if self.interactee_pred:
        #     images_in_batch = recording_utils['original_imgname']
        #     full_go = []
        #     full_bp = []
        #     full_betas = []
            
        #     #print(list_imgname)
        #     for image in images_in_batch:
        #         interactee_gt = self.interactee_pred_dict[image]
        #         global_orient_gt = interactee_gt['smpl_parameters']['global_orient']
        #         body_pose_gt = interactee_gt['smpl_parameters']['body_pose']
        #         betas_gt = interactee_gt['smpl_parameters']['betas']
        #         full_go.append(global_orient_gt)
        #         full_bp.append(body_pose_gt)
        #         full_betas.append(betas_gt)

        

        #     smpl_params_interactee = {'global_orient': np.array(full_go[:item_length]).reshape(-1,1,3),
        #                   'transl': np.array(interactee['transl']), # !!! NOT FROM EgoHMR !!!
        #                   'body_pose': np.array(full_bp[:item_length]).reshape(-1,1,69),#.reshape(-1,23,3),
        #                   'betas': np.array(full_betas[:item_length]).reshape(-1,1,10),
        #                  }
                                      
            
        # else:
        smpl_params_interactee = {'global_orient': np.array(interactee['global_orient']),
                    'transl': np.array(interactee['transl']),
                    'body_pose': np.array(interactee['body_pose']),
                    'betas': np.array(interactee['betas']),
                    }

        if self.pose_estimation_task:
            smpl_params_interactee_pe_gt = {'global_orient': np.array(interactee_gt_pose_estimation['global_orient']),
                       'transl': np.array(interactee_gt_pose_estimation['transl']),
                       'body_pose': np.array(interactee_gt_pose_estimation['body_pose']),
                       'betas': np.array(interactee_gt_pose_estimation['betas']),
                      }


        if self.global_orient_egoego or self.transl_egoego:
            pred_egoego_transl = []
            pred_egoego_go = []
            for i, img_path in enumerate(recording_utils['original_imgname']):
                try:
                    trns = self.pred_egoego[img_path]['transl']
                    gor = self.pred_egoego[img_path]['global_orient']
                except:
                    trns = pred_egoego_transl[-1] if len(pred_egoego_transl) > 0 else [0., 0., 0.]
                    gor = pred_egoego_go[-1] if len(pred_egoego_go) > 0 else [[0.,0.,0.]*3]

                pred_egoego_transl.append(trns)
                pred_egoego_go.append(gor)
            pred_egoego_transl = torch.tensor(np.array(pred_egoego_transl))

            pred_egoego_go = torch.tensor(np.array(pred_egoego_go))
            pred_egoego_global_orient = rotmat2aa(pred_egoego_go.reshape(-1, 3, 3))


            

        smpl_params_wearer = {'global_orient': np.array(wearer['global_orient']),
                              'transl': np.array(wearer['transl']),
                                'body_pose': np.array(wearer['body_pose']),
                                'betas': np.array(wearer['betas']),
                                }
        
        if self.data_type == 'rot6d':
            out_rot_wear_bp = aa_to_rotmat(torch.tensor(smpl_params_wearer['body_pose'].reshape(-1,23,3)).reshape(-1, 3)).view(-1, 23, 3, 3)
            out_rot_wear_go = aa_to_rotmat(torch.tensor(smpl_params_wearer['global_orient'].reshape(-1,1,3)).reshape(-1, 3)).view(-1, 1, 3, 3)

            out_rot_int_bp = aa_to_rotmat(torch.tensor(smpl_params_interactee['body_pose'].reshape(-1,23,3)).reshape(-1, 3)).view(-1, 23, 3, 3)
            out_rot_int_go = aa_to_rotmat(torch.tensor(smpl_params_interactee['global_orient'].reshape(-1,1,3)).reshape(-1, 3)).view(-1, 1, 3, 3)

            out_6d_wear_bp = rotmat_to_rot6d(out_rot_wear_bp.reshape(-1, 3, 3)).reshape(-1, 1, 23*6)
            out_6d_wear_go = rotmat_to_rot6d(out_rot_wear_go.reshape(-1, 3, 3)).reshape(-1, 1, 6)

            out_6d_int_bp = rotmat_to_rot6d(out_rot_int_bp.reshape(-1, 3, 3)).reshape(-1, 1, 23*6)
            out_6d_int_go = rotmat_to_rot6d(out_rot_int_go.reshape(-1, 3, 3)).reshape(-1, 1, 6)

            smpl_params_interactee['body_pose'] = out_6d_int_bp.numpy()
            smpl_params_interactee['global_orient'] = out_6d_int_go.numpy()
            smpl_params_wearer['body_pose'] = out_6d_wear_bp.numpy()
            smpl_params_wearer['global_orient'] = out_6d_wear_go.numpy()


            '''
            wear_bp = rotmat_to_rot6d(aa_to_rotmat(torch.tensor(smpl_params_wearer['body_pose']).reshape(-1, 3)).reshape(-1, 23, 3, 3)).reshape(-1, 1, 23*6)
            wear_go = rotmat_to_rot6d(aa_to_rotmat(torch.tensor(smpl_params_wearer['global_orient']).reshape(-1, 3)).reshape(-1, 1, 3, 3)).reshape(-1, 1, 6)

            interactee_bp = rotmat_to_rot6d(aa_to_rotmat(torch.tensor(smpl_params_interactee['body_pose']).reshape(-1, 3)).reshape(-1, 23, 3, 3)).reshape(-1, 1, 23*6)
            interactee_go = rotmat_to_rot6d(aa_to_rotmat(torch.tensor(smpl_params_interactee['global_orient']).reshape(-1, 3)).reshape(-1, 1, 3, 3)).reshape(-1, 1, 6)
            
            smpl_params_interactee['body_pose'] = interactee_bp.numpy()
            smpl_params_interactee['global_orient'] = interactee_go.numpy()
            smpl_params_wearer['body_pose'] = wear_bp.numpy()
            smpl_params_wearer['global_orient'] = wear_go.numpy()'''
        
        elif self.data_type == 'angle':
            # Nedd to # apply the global rotation to the global orientation
            global_orient_int = smpl_params_interactee['global_orient']
            global_orient_wear = smpl_params_wearer['global_orient']


            

            #for i in range(global_orient_int.shape[0]):
            #    global_orient_int[i] = rot_aa(global_orient_int[i].reshape(-1, 3), rot=0.).reshape(-1, 1, 3)
            #    global_orient_wear[i] = rot_aa(global_orient_wear[i].reshape(-1, 3), rot=0.).reshape(-1, 1, 3)
            

            smpl_params_interactee['global_orient'] = global_orient_int
            smpl_params_wearer['global_orient'] = global_orient_wear

            if self.pose_estimation_task:
                global_orient_int_pe_gt = smpl_params_interactee_pe_gt['global_orient']
                smpl_params_interactee_pe_gt['global_orient'] = global_orient_int_pe_gt
           


        has_smpl_params = {'global_orient': True,
                           'transl': True,
                           'body_pose': True,
                           'betas': True
                           }
        
        smpl_params_is_axis_angle = {'global_orient': True,
                                     'transl': False,
                                     'body_pose': True,
                                     'betas': False
                                    }


        do_like_egoego=False
        normalize_altogether=False
        if normalize_altogether:
            mean = self.mean[0, :self.numdims]
            std = self.std[0, :self.numdims]
            full_pose_aa_interactee = np.concatenate([smpl_params_interactee['global_orient'], 
                                                    smpl_params_interactee['body_pose']], axis=-1)
            full_pose_aa_interactee = (full_pose_aa_interactee.reshape(self.motion_length, -1) - mean) / std
            

            full_pose_aa_wearer = np.concatenate([smpl_params_wearer['global_orient'],
                                                    smpl_params_wearer['body_pose']], axis=-1)
            full_pose_aa_wearer = (full_pose_aa_wearer.reshape(self.motion_length, -1) - mean) / std
            
            interactee = torch.tensor(full_pose_aa_interactee, dtype=torch.float32).unsqueeze(1)
            wearer = torch.tensor(full_pose_aa_wearer, dtype=torch.float32).unsqueeze(1)

            motion = torch.cat([wearer, interactee], dim=1)
        
        else:
            
            full_pose_aa_interactee = smpl_params_interactee['body_pose']
            full_pose_aa_wearer = smpl_params_wearer['body_pose']

            if self.pose_estimation_task:
                full_pose_aa_interactee_pe_gt = smpl_params_interactee_pe_gt['body_pose']



            mean = self.mean[0, self.go_dims:self.numdims] 
            std = self.std[0, self.go_dims:self.numdims]

            
            if item_length!=self.motion_length:
                length_to_pad = self.motion_length - item_length
                full_pose_aa_interactee = np.concatenate([full_pose_aa_interactee, np.zeros((length_to_pad, 1, 69))], axis=0)
                full_pose_aa_wearer = np.concatenate([full_pose_aa_wearer, np.zeros((length_to_pad, 1, 69))], axis=0)
                if self.pose_estimation_task:
                    full_pose_aa_interactee_pe_gt = np.concatenate([full_pose_aa_interactee_pe_gt, np.zeros((length_to_pad, 1, 69))], axis=0)

            full_pose_aa_interactee = (full_pose_aa_interactee.reshape(self.motion_length, -1) - mean) / std
            full_pose_aa_wearer = (full_pose_aa_wearer.reshape(self.motion_length, -1) - mean) / std
            if self.pose_estimation_task:
                full_pose_aa_interactee_pe_gt = (full_pose_aa_interactee_pe_gt.reshape(self.motion_length, -1) - mean) / std
            
            interactee = torch.tensor(full_pose_aa_interactee, dtype=torch.float32).unsqueeze(1)
            wearer = torch.tensor(full_pose_aa_wearer, dtype=torch.float32).unsqueeze(1)

            if self.pose_estimation_task:
                interactee_pe_gt = torch.tensor(full_pose_aa_interactee_pe_gt, dtype=torch.float32).unsqueeze(1)

            motion = torch.cat([wearer, interactee], dim=1)

            #! NOT NORMALIZED
            interactee_go = torch.tensor(smpl_params_interactee['global_orient'], dtype=torch.float32)
            wearer_go = torch.tensor(smpl_params_wearer['global_orient'], dtype=torch.float32)

            if self.pose_estimation_task:
                interactee_go_pe_gt = torch.tensor(smpl_params_interactee_pe_gt['global_orient'], dtype=torch.float32)

            if item_length!=self.motion_length:
                interactee_go = torch.cat([interactee_go, torch.zeros((length_to_pad, 1, 3))], dim=0)
                wearer_go = torch.cat([wearer_go, torch.zeros((length_to_pad, 1, 3))], dim=0)

                if self.pose_estimation_task:
                    interactee_go_pe_gt = torch.cat([interactee_go_pe_gt, torch.zeros((length_to_pad, 1, 3))], dim=0)

            
            mean = self.mean[0, :self.go_dims]
            std = self.std[0, :self.go_dims]
            interactee_go = (interactee_go - mean) / std
            wearer_go = (wearer_go - mean) / std

            if self.pose_estimation_task:
                interactee_go_pe_gt = (interactee_go_pe_gt - mean) / std

            
            global_orient = torch.cat([wearer_go, interactee_go], dim=1)
            
            motion = torch.cat([global_orient, motion], dim=-1)
            if self.pose_estimation_task:
                motion_interacte_pe_gt = torch.cat([interactee_go_pe_gt, interactee_pe_gt], dim=-1)

        utils = torch.cat([fx, cx, cy, center, scale], dim=1)
        if item_length!=self.motion_length:
            utils = torch.cat([utils, torch.zeros((length_to_pad, 6))], dim=0)

                
        #betas = torch.cat([torch.tensor(smpl_params_wearer['betas'], dtype=torch.float32).unsqueeze(0),
        #                    torch.tensor(smpl_params_interactee['betas'], dtype=torch.float32).unsqueeze(0)], dim=0)

        interactee_transl = torch.tensor(smpl_params_interactee['transl'], dtype=torch.float32)
        wearer_transl = torch.tensor(smpl_params_wearer['transl'], dtype=torch.float32)

        if self.pose_estimation_task:
            interactee_transl_pe_gt = torch.tensor(smpl_params_interactee_pe_gt['transl'], dtype=torch.float32)

        if item_length!=self.motion_length:
            interactee_transl = torch.cat([interactee_transl, torch.zeros((length_to_pad, 1, 3))], dim=0)
            wearer_transl = torch.cat([wearer_transl, torch.zeros((length_to_pad, 1, 3))], dim=0)
            if self.pose_estimation_task:
                interactee_transl_pe_gt = torch.cat([interactee_transl_pe_gt, torch.zeros((length_to_pad, 1, 3))], dim=0)


        if self.predict_transl and self.data_type == 'angle':
            interactee_transl = (interactee_transl - self.mean[0,-3:]) / self.std[0,-3:]
            wearer_transl = (wearer_transl - self.mean[0,-3:]) / self.std[0,-3:]
            if self.pose_estimation_task:
                interactee_transl_pe_gt = (interactee_transl_pe_gt - self.mean[0, self.numdims:self.numdims+3]) / self.std[0, self.numdims:self.numdims+3]

        transl = torch.cat([wearer_transl, interactee_transl], dim=1).permute(1, 0, 2)
        if self.pose_estimation_task:
            interactee_transl_pe_gt = interactee_transl_pe_gt.permute(1, 0, 2)

        interactee_beta = torch.tensor(smpl_params_interactee['betas'], dtype=torch.float32)
        wearer_beta = torch.tensor(smpl_params_wearer['betas'], dtype=torch.float32)
        if self.pose_estimation_task:
            interactee_beta_pe_gt = torch.tensor(smpl_params_interactee_pe_gt['betas'], dtype=torch.float32)

        if item_length!=self.motion_length:
            interactee_beta = torch.cat([interactee_beta, torch.zeros((length_to_pad, 1, 10))], dim=0)
            wearer_beta = torch.cat([wearer_beta, torch.zeros((length_to_pad, 1, 10))], dim=0)
            if self.pose_estimation_task:
                interactee_beta_pe_gt = torch.cat([interactee_beta_pe_gt, torch.zeros((length_to_pad, 1, 10))], dim=0).permute(1, 0, 2)

        beta = torch.cat([wearer_beta, interactee_beta], dim=1).permute(1, 0, 2)

        length = torch.tensor([item_length], dtype=torch.int32)
        
        
        # if 'image' in self.condition or self.pred_betas:
        #     images = []
        #     number_of_images = len(recording_utils['original_imgname'])
        #     image_idx = np.random.randint(0, number_of_images)
        #     img_path = os.path.join('./datasets/EgoBody/', recording_utils['original_imgname'][image_idx])
        #     cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        #     img_height, img_width, img_channels = cvimg.shape
        #     width, height = 224, 224
            
        #     bbox_size = recording_utils['scale'][image_idx]*200
        #     center_x = recording_utils['center'][image_idx][0] + bbox_size
        #     center_y = recording_utils['center'][image_idx][1] + bbox_size

        #     img_patch_cv, trans_crop = generate_image_patch(cvimg,
        #                                     center_x, center_y,
        #                                     bbox_size, bbox_size,
        #                                     patch_width=width, patch_height=height,
        #                                     do_flip=False, scale=1.0, rot=0.)
        #     image = img_patch_cv.copy()

            
        #     image = image[:, :, ::-1]  # [224, 224, 3] BGR-->RGB

        #     img_patch_cv = image.copy()
        #     img_patch = convert_cvimg_to_tensor(image)

        #     # apply RGB normalization
        #     color_scale = [1.0, 1.0, 1.0]
        #     mean_col, std_col = 255.*np.array([0.485, 0.456, 0.406]), 255. *np.array([0.229, 0.224, 0.225])
        #     for n_c in range(img_channels):
        #         img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)
        #         if mean is not None and std is not None:
        #             img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean_col[n_c]) / std_col[n_c]
        #     images = torch.tensor(img_patch, dtype=torch.float32)
        #     '''
        #     for i, img_path in enumerate(recording_utils['original_imgname']):
        #         img_path = os.path.join('./datasets/EgoBody/', img_path)
        #         cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        #         img_height, img_width, img_channels = cvimg.shape
        #         width, height = 224, 224
                
        #         bbox_size = recording_utils['scale'][i]*200
        #         center_x = recording_utils['center'][i][0] + bbox_size
        #         center_y = recording_utils['center'][i][1] + bbox_size

        #         img_patch_cv, trans_crop = generate_image_patch(cvimg,
        #                                         center_x, center_y,
        #                                         bbox_size, bbox_size,
        #                                         patch_width=width, patch_height=height,
        #                                         do_flip=False, scale=1.0, rot=0.)
        #         image = img_patch_cv.copy()

                
        #         image = image[:, :, ::-1]  # [224, 224, 3] BGR-->RGB

        #         img_patch_cv = image.copy()
        #         img_patch = convert_cvimg_to_tensor(image)

        #         # apply RGB normalization
        #         color_scale = [1.0, 1.0, 1.0]
        #         mean_col, std_col = 255.*np.array([0.485, 0.456, 0.406]), 255. *np.array([0.229, 0.224, 0.225])
        #         for n_c in range(img_channels):
        #             img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)
        #             if mean is not None and std is not None:
        #                 img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean_col[n_c]) / std_col[n_c]
        #         images.append(img_patch)
        #     images = np.stack(images, axis=0)
        #     images = torch.tensor(images, dtype=torch.float32)

        #     nuber_of_images = images.shape[0]
        #     # Sample an image
        #     image_idx = np.random.randint(0, nuber_of_images)
        #     images = images[image_idx]'''


        
                

        
        if 'scene' in self.condition and 'image' not in self.condition:
            #if self.pred_betas:
            #    return motion, transl, beta, utils, scene_pcd_verts, images
            #else:
            #    return motion, transl, beta, utils, scene_pcd_verts
            
            if self.transl_egoego:
                return motion, transl, beta, utils, scene_pcd_verts, pred_egoego_transl, pred_egoego_global_orient, length
            else:
                if self.pose_estimation_task:
                    return motion, transl, beta, utils, scene_pcd_verts, length 
                else:
                    return motion, transl, beta, utils, scene_pcd_verts, length, list_imgname

        # elif 'scene' in self.condition and 'image' in self.condition:
            
        #     return motion, transl, beta, utils, scene_pcd_verts, images, length
        # elif 'scene' not in self.condition and 'image' in self.condition:
        #     return motion, transl, beta, utils, images, length
        else:
            return motion, transl, beta, utils, length

class EgoBodyData2(data.Dataset):

    def __init__(
            self,
            mean, 
            std,
            split_file,
            motion_dir,
            #max_motion_length,
            #min_motion_length,
            tiny=False,
            debug=False,
            progress_bar=True,
            **kwargs,
    ):
        
        #self.max_motion_length = max_motion_length
        #self.min_motion_length = min_motion_length

        self.motion_length = 20 

        split = split_file.split("/")[-1].split(".")[0]
        motion_dir = pjoin(motion_dir, split)

        self.do_augment = False #True if split == 'train' else False



        self.mean = np.load('./datasets/EgoBody/our_process_smpl/mean.npy') #mean
        self.std = np.load('./datasets/EgoBody/our_process_smpl/std.npy') #std
        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.id_list = id_list

        #id_list = os.listdir(motion_dir)

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10


        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading EgoBody {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)

        count = 0
        bad_count = 0
        new_name_list = []
        length_list = []
        for i, name in enumerator:
            if count > maxdata:
                break
            try:
                motion = np.load(pjoin(motion_dir, name+'.npy'), allow_pickle=True).item()
                if len(motion['interactee']['global_orient']) < self.motion_length:
                    continue

                flag = False
                data_dict[name] = {
                    "video": motion['video'],
                    "recording_utils": motion['recording_utils'],
                    "interactee": motion['interactee'],
                    'wearer': motion['wearer'],
                }
                new_name_list.append(name)
                length_list.append(len(motion))
                count += 1
            except:
                pass
        
        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = 144 #motion.shape[1]
        self.name_list = name_list

    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, item):
        idx = item
        data = self.data_dict[self.name_list[idx]]
        interactee = data["interactee"]
        wearer = data["wearer"]

        # Sample idx for motion
        m_length = len(interactee['global_orient'])
        start_idx = np.random.randint(0, m_length - self.motion_length)
        end_idx = start_idx + self.motion_length
        
        smpl_params_interactee = {'global_orient': np.array(interactee['global_orient'])[start_idx:end_idx],
                       'transl': np.array(interactee['transl'])[start_idx:end_idx],
                       'body_pose': np.array(interactee['body_pose'])[start_idx:end_idx],
                       'betas': np.array(interactee['betas'])[start_idx:end_idx],
                      }

        smpl_params_wearer = {'global_orient': np.array(wearer['global_orient'])[start_idx:end_idx],
                              'transl': np.array(wearer['transl'])[start_idx:end_idx],
                                'body_pose': np.array(wearer['body_pose'])[start_idx:end_idx],
                                'betas': np.array(wearer['betas'])[start_idx:end_idx],
                                }
        

        has_smpl_params = {'global_orient': True,
                           'transl': True,
                           'body_pose': True,
                           'betas': True
                           }
        
        smpl_params_is_axis_angle = {'global_orient': True,
                                     'transl': False,
                                     'body_pose': True,
                                     'betas': False
                                    }

        
        if self.do_augment:
            auge_scale, rot, do_flip, color_scale, tx, ty = do_augmentation(augm_config)
        else:
            auge_scale, rot, do_flip, color_scale, tx, ty = 1.0, 0, False, [1.0, 1.0, 1.0], 0., 0.

        smpl_params_interactee, has_smpl_params_interactee = smpl_param_processing(smpl_params_interactee, has_smpl_params, rot, do_flip, self.motion_length)
        smpl_params_wearer, has_smpl_params_wearer = smpl_param_processing(smpl_params_wearer, has_smpl_params, rot, do_flip, self.motion_length)

        full_pose_aa_interactee = np.concatenate([smpl_params_interactee['global_orient'], 
                                                  smpl_params_interactee['body_pose']], axis=-1).reshape(self.motion_length,-1, 3)
        
        full_pose_aa_wearer = np.concatenate([smpl_params_wearer['global_orient'],
                                                smpl_params_wearer['body_pose']], axis=-1).reshape(self.motion_length,-1, 3)
        
        full_pose_aa_interactee = torch.tensor(full_pose_aa_interactee, dtype=torch.float32)#.unsqueeze(0)
        full_pose_aa_wearer = torch.tensor(full_pose_aa_wearer, dtype=torch.float32)#.unsqueeze(0)

        # Rescale according to camera
        #trans_int = torch.tensor(smpl_params_interactee['transl'], dtype=torch.float32).unsqueeze(0)
        #trans_wea = torch.tensor(smpl_params_wearer['transl'], dtype=torch.float32).unsqueeze(0)

        full_pose_rot6d_interactee = []
        full_pose_rot6d_wearer = []
        for i in range(self.motion_length):
            full_pose_rotmat_interactee = aa_to_rotmat(full_pose_aa_interactee[i]).view(1, -1, 3, 3)
            full_pose_rotmat_wearer = aa_to_rotmat(full_pose_aa_wearer[i]).view(1, -1, 3, 3)

            rot6d_int = rotmat_to_rot6d(full_pose_rotmat_interactee.reshape(-1, 3, 3), rot6d_mode='diffusion').reshape(-1, 6)
            rot6d_wea = rotmat_to_rot6d(full_pose_rotmat_wearer.reshape(-1, 3, 3), rot6d_mode='diffusion').reshape(-1, 6)


            full_pose_rot6d_interactee.append(rot6d_int)
            full_pose_rot6d_wearer.append(rot6d_wea)
        
        full_pose_rot6d_interactee = torch.stack(full_pose_rot6d_interactee, dim=0).reshape(self.motion_length, -1)
        full_pose_rot6d_wearer = torch.stack(full_pose_rot6d_wearer, dim=0).reshape(self.motion_length, -1)

        
        
        # Z normalization
        full_pose_rot6d_interactee = (full_pose_rot6d_interactee - self.mean) / self.std
        full_pose_rot6d_wearer = (full_pose_rot6d_wearer - self.mean) / self.std

        interactee = full_pose_rot6d_interactee.unsqueeze(1)
        wearer = full_pose_rot6d_wearer.unsqueeze(1)

        motion = torch.cat([wearer, interactee], dim=1)
        #betas = torch.cat([torch.tensor(smpl_params_wearer['betas'], dtype=torch.float32).unsqueeze(0),
        #                    torch.tensor(smpl_params_interactee['betas'], dtype=torch.float32).unsqueeze(0)], dim=0)


        #m_length = len(motion)
        return motion #, betas # n_frames, actor, 144
    
    def renorm(self, data):
        return data * self.std + self.mean



class DatasetEgobody(data.Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 dataset_file: str,
                 data_root: str,
                 train: bool = True,
                 split='train',
                 spacing=1,
                 add_scale=1.0,
                 device=None,
                 do_augment=False,
                 scene_type='none', #'whole_scene',
                 scene_cano=False,
                 scene_downsample_rate=1,
                 get_diffuse_feature=False,
                 body_rep_stats_dir='',
                 load_stage1_transl=False,
                 stage1_result_path='',
                 scene_crop_by_stage1_transl=False,
                 **kwargs,
                 ):
        """
        Dataset class used for loading images and corresponding annotations.
        """
        super(DatasetEgobody, self).__init__()
        self.train = train
        self.split = split
        self.cfg = cfg
        self.device = device
        self.do_augment = do_augment

        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        self.fx_norm_coeff = self.cfg.CAM.FX_NORM_COEFF
        self.fy_norm_coeff = self.cfg.CAM.FY_NORM_COEFF
        self.cx_norm_coeff = self.cfg.CAM.CX_NORM_COEFF
        self.cy_norm_coeff = self.cfg.CAM.CY_NORM_COEFF

        self.data_root = data_root
        dataset_file = os.path.join(self.data_root, dataset_file)

        self.data = np.load(dataset_file)
        #!! with open(os.path.join(self.data_root, 'transf_matrices_all_seqs.pkl'), 'rb') as fp:
        #!!    self.transf_matrices = pkl.load(fp)

        self.imgname = self.data['imgname']

        [self.imgname, self.seq_names, _] = zip(*[get_right_full_img_pth(x, self.data_root) for x in self.imgname])   # absolute dir
        self.seq_names = [basename(x) for x in self.seq_names][::spacing]
        self.imgname = self.imgname[::spacing]

        body_permutation_2d = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]  # for openpose 25 topology
        body_permutation_3d = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]  # for smpl 24 topology
        self.flip_2d_keypoint_permutation = body_permutation_2d
        self.flip_3d_keypoint_permutation = body_permutation_3d

        # Bounding boxes are assumed to be in the center and scale format
        self.center = self.data['center'][::spacing]
        self.scale = self.data['scale'][::spacing] * add_scale

        self.has_smpl = np.ones(len(self.imgname))
        self.body_pose = self.data['pose'].astype(np.float)[::spacing]  # [n_sample, 69]
        self.betas = self.data['shape'].astype(np.float)[::spacing]
        self.global_orient_pv = self.data['global_orient_pv'].astype(np.float)[::spacing]  # [n_sample, 3]
        self.transl_pv = self.data['transl_pv'].astype(np.float)[::spacing]

        self.cx = self.data['cx'].astype(np.float)[::spacing]
        self.cy = self.data['cy'].astype(np.float)[::spacing]
        self.fx = self.data['fx'].astype(np.float)[::spacing]
        self.fy = self.data['fy'].astype(np.float)[::spacing]


        keypoints_openpose = self.data['valid_keypoints'][::spacing]
        self.keypoints_2d = keypoints_openpose
        self.keypoints_3d_pv = self.data['3d_joints_pv'].astype(np.float)[::spacing]

        # Get gender data, if available
        gender = self.data['gender'][::spacing]
        self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)

        self.load_stage1_transl = load_stage1_transl
        if self.load_stage1_transl:
            with open(stage1_result_path, 'rb') as fp:
                stage1_result = pkl.load(fp)
            self.stage1_transl_full = stage1_result['pred_cam_full_list'].astype(np.float)[::spacing]  # [n_samples, 3]

        ######## get mean/var for body representation feature in EgoHMR(to normalize for diffusion model)
        if get_diffuse_feature and split == 'train' and self.train:
            # 144-d
            global_orient_pv_all = torch.from_numpy(self.global_orient_pv).float()
            body_pose_all = torch.from_numpy(self.body_pose).float()
            full_pose_aa_all = torch.cat([global_orient_pv_all, body_pose_all], dim=1).reshape(-1, 24, 3)  # [n, 24, 3]
            full_pose_rotmat_all = aa_to_rotmat(full_pose_aa_all.reshape(-1, 3)).view(-1, 24, 3, 3)  # [bs, 24, 3, 3]
            full_pose_rot6d_all = rotmat_to_rot6d(full_pose_rotmat_all.reshape(-1, 3, 3),
                                                  rot6d_mode='diffusion').reshape(-1, 24, 6).reshape(-1, 24 * 6)  # [n, 144]
            full_pose_rot6d_all = full_pose_rot6d_all.detach().cpu().numpy()
            Xmean = full_pose_rot6d_all.mean(axis=0)  # [d]
            Xstd = full_pose_rot6d_all.std(axis=0)  # [d]
            stats_root = os.path.join(body_rep_stats_dir, 'preprocess_stats')
            os.makedirs(stats_root) if not os.path.exists(stats_root) else None
            Xstd[0:6] = Xstd[0:6].mean() / 1.0  # for global orientation
            Xstd[6:] = Xstd[6:].mean() / 1.0  # for body pose
            np.savez_compressed(os.path.join(stats_root, 'preprocess_stats.npz'), Xmean=Xmean, Xstd=Xstd)
            print('[INFO] mean/std for body_rep saved.')


        self.smpl_male = smplx.create('datasets/data/smpl', model_type='smpl', gender='male')
        self.smpl_female = smplx.create('datasets/data/smpl', model_type='smpl', gender='female')

        self.dataset_len = len(self.imgname)
        print('[INFO] find {} samples in {}.'.format(self.dataset_len, dataset_file))

        ########### read scene pcd
        self.scene_type = scene_type
        # self.scene_cube_normalize = scene_cube_normalize
        if self.scene_type == 'whole_scene':
            with open(os.path.join(self.data_root, 'Egohmr_scene_preprocess_s1_release/pcd_verts_dict_{}.pkl'.format(split)), 'rb') as f:
                self.pcd_verts_dict_whole_scene = pkl.load(f)
            with open(os.path.join(self.data_root, 'Egohmr_scene_preprocess_s1_release/map_dict_{}.pkl'.format(split)), 'rb') as f:
                self.pcd_map_dict_whole_scene = pkl.load(f)
        elif self.scene_type == 'cube':
            if not scene_crop_by_stage1_transl:
                self.pcd_root = os.path.join(self.data_root, 'Egohmr_scene_preprocess_cube_s2_from_gt_release')
            else:
                self.pcd_root = os.path.join(self.data_root, 'Egohmr_scene_preprocess_cube_s2_from_pred_release')
        #!else:
        #!    print('[ERROR] wrong scene_type!')
        #!    exit()


        df = pd.read_csv(os.path.join(self.data_root, 'data_info_release.csv'))
        recording_name_list = list(df['recording_name'])
        scene_name_list = list(df['scene_name'])
        self.scene_name_dict = dict(zip(recording_name_list, scene_name_list))
        self.add_trans = np.array([[1.0, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.scene_cano = scene_cano
        self.scene_downsample_rate = scene_downsample_rate

        self.nfeats = 144



    def get_transf_matrices_per_frame(self, img_name, seq_name):
        transf_mtx_seq = self.transf_matrices[seq_name]
        kinect2holo = transf_mtx_seq['trans_kinect2holo'].astype(np.float32)  # [4,4], one matrix for all frames in the sequence
        holo2pv_dict = transf_mtx_seq['trans_world2pv']  # a dict, # frames items, each frame is a 4x4 matrix
        timestamp = basename(img_name).split('_')[0]
        holo2pv = holo2pv_dict[str(timestamp)].astype(np.float32)
        return kinect2holo, holo2pv



    def __len__(self) -> int:
        return len(self.scale)

    def __getitem__(self, idx: int) -> Dict:
        self.data_root = './datasets/EgoBody'
        image_file = os.path.join(self.data_root, self.imgname[idx])  # absolute path

        seq_name = self.seq_names[idx]
        keypoints_2d = self.keypoints_2d[idx].copy()  # [25, 3], openpose joints
        keypoints_3d = self.keypoints_3d_pv[idx][0:24].copy()  # [24, 3], smpl joints

        center = self.center[idx].copy().astype(np.float32)
        center_x = center[0]
        center_y = center[1]
        bbox_size = self.scale[idx].astype(np.float32) * 200
        body_pose = self.body_pose[idx].copy().astype(np.float32)  # 69
        betas = self.betas[idx].copy().astype(np.float32)  # [10]
        global_orient = self.global_orient_pv[idx].copy().astype(np.float32)  # 3
        transl = self.transl_pv[idx].copy().astype(np.float32)  # 3
        gender = self.gender[idx].copy()

        fx = self.fx[idx].copy()
        fy = self.fy[idx].copy()
        cx = self.cx[idx].copy()
        cy = self.cy[idx].copy()

        smpl_params = {'global_orient': global_orient,
                       'transl': transl,
                       'body_pose': body_pose,
                       'betas': betas
                      }
        has_smpl_params = {'global_orient': True,
                           'transl': True,
                           'body_pose': True,
                           'betas': True
                           }
        smpl_params_is_axis_angle = {'global_orient': True,
                                     'transl': False,
                                     'body_pose': True,
                                     'betas': False
                                    }

        item = {}
        #!! item['transf_kinect2holo'], item['transf_holo2pv'] = self.get_transf_matrices_per_frame(image_file, seq_name)

        #!!pcd_trans_kinect2pv = np.matmul(item['transf_holo2pv'], item['transf_kinect2holo'])
        #!!pcd_trans_kinect2pv = np.matmul(self.add_trans, pcd_trans_kinect2pv)
        #!!temp = "/".join(image_file.split('/')[-5:])
        if self.scene_type == 'whole_scene':
            scene_pcd_verts = self.pcd_verts_dict_whole_scene[self.pcd_map_dict_whole_scene[temp]]  # [20000, 3], in kinect main coord
            scene_pcd_verts = points_coord_trans(scene_pcd_verts, pcd_trans_kinect2pv)
        elif self.scene_type == 'cube':
            recording_name = image_file.split('/')[-4]
            img_name = image_file.split('/')[-1]
            scene_pcd_path = os.path.join(self.pcd_root, self.split, recording_name, image_file.split('/')[-3], img_name[:-3]+'npy')
            scene_pcd_verts = np.load(scene_pcd_path)  # in scene coord
            # transformation from master kinect RGB camera to scene mesh
            calib_trans_dir = os.path.join(self.data_root, 'calibrations', recording_name)
            cam2world_dir = os.path.join(calib_trans_dir, 'cal_trans/kinect12_to_world')
            with open(os.path.join(cam2world_dir, self.scene_name_dict[recording_name] + '.json'), 'r') as f:
                trans_scene_to_main = np.array(json.load(f)['trans'])
            trans_scene_to_main = np.linalg.inv(trans_scene_to_main)
            pcd_trans_scene2pv = np.matmul(pcd_trans_kinect2pv, trans_scene_to_main)
            scene_pcd_verts = points_coord_trans(scene_pcd_verts, pcd_trans_scene2pv)  # nowall: 5000, withwall: 5000+30*30*5=9500

        #################################### data augmentation
        augm_config = False #!self.cfg.DATASETS.CONFIG
        img_patch, keypoints_2d_crop_auge, keypoints_2d_vis_mask, keypoints_2d_full_auge, \
                scene_pcd_verts_full_auge, keypoints_3d_crop_auge, keypoints_3d_full_auge, smpl_params, has_smpl_params, \
                center_x_auge, center_y, cam_cx_auge, auge_scale, rotated_img \
                = get_example(image_file if self.scene_type != 'none' else None,
                            center_x, center_y, bbox_size, bbox_size,
                            keypoints_2d, keypoints_3d, smpl_params, has_smpl_params,
                            self.flip_2d_keypoint_permutation, self.flip_3d_keypoint_permutation,
                            self.img_size, self.img_size, self.mean, self.std,
                            self.do_augment, augm_config,
                            fx, cam_cx=cx, cam_cy=cy,
                            scene_pcd_verts=scene_pcd_verts if self.scene_type != 'none' else None,
                            smpl_male=self.smpl_male, smpl_female=self.smpl_female, gender=gender)
            
        if self.scene_type != 'none':
            item['img'] = img_patch
            item['imgname'] = image_file
            item['orig_img'] = rotated_img  # original img rotate around (center_x_auge, center_y_auge)
            ###### 2d joints
            item['keypoints_2d'] = keypoints_2d_crop_auge.astype(np.float32)  # [25, 3]
            item['orig_keypoints_2d'] = keypoints_2d_full_auge.astype(np.float32)
            item['keypoints_2d_vis_mask'] = keypoints_2d_vis_mask  # [25] vis mask for openpose joint in augmented cropped img

            ###### 3d joints
            item['keypoints_3d'] = keypoints_3d_crop_auge.astype(np.float32)  # [24, 3]
            item['keypoints_3d_full'] = keypoints_3d_full_auge.astype(np.float32)

            ###### camera params
            item['fx'] = (fx / self.fx_norm_coeff).astype(np.float32)
            item['fy'] = (fy / self.fy_norm_coeff).astype(np.float32)
            item['cam_cx'] = cam_cx_auge.astype(np.float32)
            item['cam_cy'] = cy.astype(np.float32)
            ###### bbox params
            item['box_center'] = np.array([center_x_auge, center_y]).astype(np.float32)
            item['box_size'] = (bbox_size * auge_scale).astype(np.float32)

        ###### smpl params
        item['smpl_params'] = smpl_params
        for key in item['smpl_params'].keys():
            item['smpl_params'][key] = item['smpl_params'][key].astype(np.float32)
        item['has_smpl_params'] = has_smpl_params
        item['smpl_params_is_axis_angle'] = smpl_params_is_axis_angle
        # item['idx'] = idx
        item['gender'] = gender

        ###### scene verts
        #!scene_pcd_verts_full_auge = scene_pcd_verts_full_auge.astype(np.float32)  # [n_pts, 3]
        #!scene_pcd_verts_full_auge = scene_pcd_verts_full_auge[::self.scene_downsample_rate]
        #!item['scene_pcd_verts_full'] = scene_pcd_verts_full_auge  # [20000, 3]
        # only for test
        if self.load_stage1_transl:
            item['stage1_transl_full'] = self.stage1_transl_full[idx].astype(np.float32)

        
        return item