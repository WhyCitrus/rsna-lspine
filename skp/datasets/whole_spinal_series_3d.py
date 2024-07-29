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

        self.inputs = df[self.cfg.inputs].tolist()
        self.labels = df[self.cfg.targets].values 
        self.slice_start = df.slice_start.values
        self.slice_end = df.slice_end.values

        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self):
        return len(self.inputs) 

    def load_stack(self, i):
        # assumes that inputs are list of directories
        # where each directory contains all the images in a stack as PNG 
        # and that filenames are sortable

        # we are going to use the 3ch images but only take middle slice
        assert self.cfg.cv2_load_flag == cv2.IMREAD_COLOR
        images = np.sort(glob.glob(os.path.join(self.cfg.data_dir, self.inputs[i], "*.png")))[self.slice_start[i]:self.slice_end[i]]
        x = np.stack([cv2.imread(im, self.cfg.cv2_load_flag) for im in images], axis=0)
        if len(x) < 12:
            pad = np.expand_dims(np.zeros_like(x[0]), axis=0)
            diff = 12 - len(x)
            x = np.concatenate([x] + [pad] * diff, axis=0)
        x = x[..., 1]
        x = np.expand_dims(x, axis=-1)
        assert len(x) == 12, f"len(x) is {len(x)}"
        return x

    def get(self, i):
        try:
            x = self.load_stack(i)
            y = self.labels[i]
            return x, y
        except Exception as e:
            print("ERROR:", self.inputs[i], "\n", e)
            return None

    def __getitem__(self, i):
        data = self.get(i)
        while not isinstance(data, tuple):
            i = np.random.randint(len(self))
            data = self.get(i)

        x, y = data

        to_transform = {"image": x[0]}
        for idx in range(1, x.shape[0]):
            to_transform.update({f"image{idx}": x[idx]})

        xt = self.transforms(**to_transform)
        x = np.stack([xt["image"]] + [xt[f"image{idx}"] for idx in range(1, x.shape[0])])
        x = torch.from_numpy(x)
        x = x.float()
        x = x.permute(3, 0, 1, 2)

        y = torch.tensor(y).float()

        return {"x": x, "y": y, "index": i}
