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
        if x.ndim == 3:
            x = np.expand_dims(x, axis=-1)
        # x.shape = (Z, H, W, C)
        return x

    def get(self, i):
        try:
            x = self.load_stack(i)
            y = self.labels[i].copy() # need to make copy to make sure original is preserved
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

        if self.mode == "train" and np.random.binomial(1, 0.5):
            # flip along slice dimension
            x = np.ascontiguousarray(x[::-1])
            # edit slice coordinates
            y[[0, 3, 6, 9, 12]] = 1 - y[[0, 3, 6, 9, 12]]

        if self.mode == "train" and np.random.binomial(1, 0.5):
            # flip up-down
            x = np.ascontiguousarray(x[:, ::-1])
            # edit y coordinates
            y[[2, 5, 8, 11, 14]] = 1 - y[[2, 5, 8, 11, 14]]

        if self.mode == "train" and np.random.binomial(1, 0.5):
            # flip left-right
            x = np.ascontiguousarray(x[:, :, ::-1])
            # edit x coordinates
            y[[1, 4, 7, 10, 13]] = 1 - y[[1, 4, 7, 10, 13]]

        to_transform = {"image": x[0]}
        for idx in range(1, x.shape[0]):
            to_transform.update({f"image{idx}": x[idx]})

        xt = self.transforms(**to_transform)
        x = np.stack([xt["image"]] + [xt[f"image{idx}"] for idx in range(1, x.shape[0])])
        x = torch.from_numpy(x)
        x = x.float()
        x = x.permute(3, 0, 1, 2)

        if y.ndim == 0:
            y = torch.tensor(y).float().unsqueeze(-1)

        return {"x": x, "y": y, "index": i}
