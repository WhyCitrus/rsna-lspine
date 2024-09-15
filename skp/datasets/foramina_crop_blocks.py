import copy
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

        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self):
        return len(self.inputs) 

    def load_stack(self, i):
        assert self.cfg.cv2_load_flag == cv2.IMREAD_GRAYSCALE
        # assumes that inputs are list of directories
        # where each directory contains all the images in a stack as PNG 
        # and that filenames are sortable
        images = np.sort(glob.glob(os.path.join(self.cfg.data_dir, self.inputs[i], "*.png")))
        if np.random.binomial(1, 0.5) and self.mode == "train":
            images = images[::-1]
        mask = [False] * len(images)
        if self.cfg.image_z < len(images):
            diff = len(images) - self.cfg.image_z
            diff = diff // 2
            images = images[diff:(diff + self.cfg.image_z)]
            mask = mask[:self.cfg.image_z]
        x = np.stack([cv2.imread(im, self.cfg.cv2_load_flag) for im in images], axis=0)
        if self.cfg.image_z > len(x):
            diff = self.cfg.image_z - len(x)
            mask = [False] * len(x) + [True] * diff
            pad = np.expand_dims(np.zeros_like(x[0]), axis=0)
            x = np.concatenate([x] + [pad] * diff)
        assert len(x) == self.cfg.image_z == len(mask), f"x {len(x)}, image_z {self.cfg.image_z}, mask {len(mask)}"
        # x.shape = (Z, H, W)
        x = np.expand_dims(x, axis=-1)
        return x, mask

    def get(self, i):
        try:
            x, mask = self.load_stack(i)
            y = copy.deepcopy(self.labels[i])
            return x, y, mask
        except Exception as e:
            print("ERROR:", self.inputs[i], "\n", e)
            return None

    def __getitem__(self, i):
        data = self.get(i)
        while not isinstance(data, tuple):
            i = np.random.randint(len(self))
            data = self.get(i)

        x, y, mask = data

        to_transform = {"image": x[0]}
        for idx in range(1, x.shape[0]):
            to_transform.update({f"image{idx}": x[idx]})

        xt = self.transforms(**to_transform)
        x = np.stack([xt["image"]] + [xt[f"image{idx}"] for idx in range(1, x.shape[0])])
        x = torch.from_numpy(x)
        x = x.float()
        # x.shape = (Z, H, W, 1)
        x = x.permute(0, 3, 1, 2)
        mask = torch.tensor(mask)

        return {"x": x, "y": y, "mask": mask, "index": i}
