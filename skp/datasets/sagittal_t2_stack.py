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
        df = df.sort_values("ImagePositionPatient0", ascending=True).reset_index(drop=True)
        if len(df) > self.cfg.max_num_images:
            # Trim off both ends equally if exceeds number of images
            diff = len(df) - self.cfg.max_num_images
            diff = diff // 2
            df = df.iloc[diff:(diff + self.cfg.max_num_images)]
        assert len(df) <= self.cfg.max_num_images
        images = df.filepath.values
        if not isinstance(self.cfg.select_image_channel, type(None)):
            images = [cv2.imread(os.path.join(self.cfg.data_dir, im), self.cfg.cv2_load_flag)[..., self.cfg.select_image_channel] for im in images]
            images = [np.expand_dims(im, axis=-1) for im in images]
        else:
            images = [cv2.imread(os.path.join(self.cfg.data_dir, im), self.cfg.cv2_load_flag) for im in images]
        labels = df[self.cfg.targets].values
        return images, labels

    def get(self, i):
        #try:
            x, y = self.load_stack(i)
            return x, y
        # except Exception as e:
        #     print("ERROR:", self.dfs[i].series_id.iloc[0], "\n", e)
        #     return None

    def __getitem__(self, i):
        data = self.get(i)
        while not isinstance(data, tuple):
            i = np.random.randint(len(self))
            data = self.get(i)

        x, y = data

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
            mask = mask + [True] * diff

        x = torch.from_numpy(x)
        x = x.permute(0, 3, 1, 2)
        x = x.float()

        y = torch.from_numpy(y)
        y = y.float()

        mask = torch.tensor(mask)

        assert len(x) == len(y) == len(mask)
        if self.mode == "train":
            assert len(x) == self.cfg.max_num_images

        return {"x": x, "y": y, "mask": mask, "index": i}
