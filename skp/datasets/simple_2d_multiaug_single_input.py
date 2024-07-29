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
            df = df.drop_duplicates().reset_index(drop=True)
            df = df[df.fold == self.cfg.fold]
            if "ignore_during_val" in df.columns:
                df = df[df.ignore_during_val == 0]
            self.transforms = self.cfg.val_transforms

        self.df = df.reset_index(drop=True)
        self.inputs = df[self.cfg.inputs].tolist()
        self.labels = df[self.cfg.targets].values 

        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self):
        return len(self.inputs) 

    def get(self, i):
        try:
            filepaths = self.inputs[i].split(",")
            x = np.stack([
                cv2.imread(os.path.join(self.cfg.data_dir, fp), self.cfg.cv2_load_flag)
                for fp in filepaths
            ])
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

        if self.cfg.channel_reverse and self.mode == "train" and bool(np.random.binomial(1, 0.5)):
            x = np.ascontiguousarray(x[..., ::-1])

        x = np.stack([self.transforms(image=_)["image"] for _ in x])

        if x.ndim == 3:
            x = np.expand_dims(x, axis=-1)

        x = x.transpose(0, 3, 1, 2) # channels-last -> channels-first
        x = torch.tensor(x).float()
        if y.ndim == 0:
            y = torch.tensor(y).float().unsqueeze(-1)

        return {"x": x, "y": y, "index": i}
