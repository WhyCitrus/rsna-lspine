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
        # assumes that inputs are list of directories
        # where each directory contains all the images in a stack as PNG 
        # and that filenames are sortable
        images = np.sort(glob.glob(os.path.join(self.cfg.data_dir, self.inputs[i], "*.png")))
        if self.cfg.image_z != len(images):
            indices = np.arange(len(images))
            indices = zoom(indices, self.cfg.image_z / len(images), order=0, prefilter=False).astype("int")
            assert len(indices) == self.cfg.image_z
            images = images[indices]
        x = np.stack([cv2.imread(im, self.cfg.cv2_load_flag) for im in images], axis=0)
        if self.cfg.cv2_load_flag == cv2.IMREAD_GRAYSCALE:
            x = np.expand_dims(x, axis=-1)
        if self.cfg.num_input_channels == 2:
            x = x[..., :2]
        # channels-last -> channels-first
        x = torch.tensor(x.transpose(3, 0, 1, 2))
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
        x = self.transforms(dict(image=x))["image"]

        x = x.float()
        if y.ndim == 0:
            y = torch.tensor(y).float().unsqueeze(-1)

        return {"x": x, "y": y, "index": i}
