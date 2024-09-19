import cv2
import glob
import numpy as np
import os
import pandas as pd
import torch

from scipy.ndimage import zoom
from torch.utils.data import Dataset as TorchDataset, default_collate


train_collate_fn = default_collate
val_collate_fn = default_collate


class Dataset(TorchDataset):

    def __init__(self, cfg, mode):
        self.cfg = cfg 
        self.mode = mode
        df = pd.read_csv(self.cfg.annotations_file)
        if self.mode == "train":
            df = df[df.fold != self.cfg.fold]
            self.transforms = self.cfg.train_transforms
        elif self.mode == "val":
            df = df[df.fold == self.cfg.fold]
            self.transforms = self.cfg.val_transforms

        self.dfs = [series_df for series_id, series_df in df.groupby("series_id")]
        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self):
        return len(self.dfs) 

    def load_stack(self, i):
        df = self.dfs[i]
        df = df.sort_values("filepath") # sorted by position
        first, last = df.instance_number.values[0], df.instance_number.values[-1]
        ascending = False if first > last else True
        # sort by instance seems to be more reliable
        # however sometimes it is top to bottom and sometimes bottom to top
        # position is always bottom to top when sorted in ascending order
        df = df.sort_values("instance_number", ascending=ascending).reset_index(drop=True)
        if len(df) > self.cfg.max_num_images:
            divisor = int(np.ceil(len(df) / self.cfg.max_num_images))
            df = df.iloc[::divisor]
        assert len(df) <= self.cfg.max_num_images
        if self.mode == "train" and np.random.binomial(1, 0.5):
            # half the time during training, randomly sample a subset
            random_subset = np.random.uniform(0.5, 1)
            random_subset = int(np.round(random_subset * len(df)))
            start_index = np.random.randint(len(df) - random_subset + 1)
            df = df.iloc[start_index:start_index + random_subset]
        if self.mode == "train" and np.random.binomial(1, 0.5):
            # reverse
            df = df.iloc[::-1]
        images = df.filepath.values
        if not isinstance(self.cfg.select_image_channel, type(None)):
            images = [cv2.imread(os.path.join(self.cfg.data_dir, im), self.cfg.cv2_load_flag)[..., self.cfg.select_image_channel] for im in images]
            images = [np.expand_dims(im, axis=-1) for im in images]
        else:
            images = [cv2.imread(os.path.join(self.cfg.data_dir, im), self.cfg.cv2_load_flag) for im in images]
        labels = df[self.cfg.targets].values
        pz = df[[f"{lvl}_pz" for lvl in ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]]].values
        return images, labels, pz

    def get(self, i):
        #try:
            x, y, z = self.load_stack(i)
            return x, y, z
        # except Exception as e:
        #     print("ERROR:", self.dfs[i].series_id.iloc[0], "\n", e)
        #     return None

    def __getitem__(self, i):
        data = self.get(i)
        while not isinstance(data, tuple):
            i = np.random.randint(len(self))
            data = self.get(i)

        x, y, pz = data

        to_transform = {"image": x[0]}
        for idx in range(1, len(x)):
            to_transform.update({f"image{idx}": x[idx]})

        xt = self.transforms(**to_transform)
        x = np.stack([xt["image"]] + [xt[f"image{idx}"] for idx in range(1, len(x))])
        # x.shape = (num_images, H, W, num_ch)
        mask = [False] * len(x)
        if self.mode == "train" and len(x) < self.cfg.max_num_images:
            diff = self.cfg.max_num_images - len(x)
            xpad = np.expand_dims(np.zeros_like(x[0]), axis=0)
            x = np.concatenate([x] + [xpad] * diff)
            ypad = np.expand_dims(np.zeros_like(y[0]), axis=0)
            y = np.concatenate([y] + [ypad] * diff)
            pzpad = np.expand_dims(np.zeros_like(pz[0]), axis=0)
            pz = np.concatenate([pz] + [pzpad] * diff)
            mask = mask + [True] * diff

        x = torch.from_numpy(x)
        x = x.permute(0, 3, 1, 2)
        x = x.float()

        y = torch.from_numpy(y)
        y = y.float()

        pz = torch.from_numpy(pz)
        pz = pz.float()

        mask = torch.tensor(mask)

        assert len(x) == len(y) == len(mask) == len(pz)
        if self.mode == "train":
            assert len(x) == self.cfg.max_num_images

        return {"x": x, "y": y, "pz": pz, "mask": mask, "index": i}
