import cv2
import numpy as np
import os
import pandas as pd
import torch

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

    def get(self, i):
        try:
            img_files = self.inputs[i].split(",")
            x = [cv2.imread(os.path.join(self.cfg.data_dir, imfi), self.cfg.cv2_load_flag) for imfi in img_files]
            if isinstance(self.cfg.select_image_channel, int):
                x = [np.expand_dims(im[..., self.cfg.select_image_channel], axis=-1) for im in x]
            keys = ["image"] + [f"image{idx+1}" for idx in range(len(x) - 1)]
            assert len(keys) == len(x)
            x = {k: v for k, v in zip(keys, x)}
            y = self.labels[i]
            return x, y
        except Exception as e:
            print(e)
            return None

    def __getitem__(self, i):
        data = self.get(i)
        while not isinstance(data, tuple):
            i = np.random.randint(len(self))
            data = self.get(i)

        x, y = data
        x = self.transforms(**x)
        x = np.concatenate([x["image"]] + [x[f"image{idx+1}"] for idx in range(len(x) - 1)], axis=-1)

        if x.ndim == 2:
            x = np.expand_dims(x, axis=-1)

        x = x.transpose(2, 0, 1) # channels-last -> channels-first
        x = torch.tensor(x).float()
        if y.ndim == 0:
            y = torch.tensor(y).float().unsqueeze(-1)

        return {"x": x, "y": y, "index": i}
