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
        self.unique_ids = df.unique_id.values
        if "sampling_weight" in df.columns:
            self.sampling_weights = df.sampling_weight.values
            
        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self):
        return len(self.inputs) 

    def get(self, i):
        try:
            files = self.inputs[i].split(",")
            x1 = cv2.imread(os.path.join(self.cfg.data_dir, files[0]), self.cfg.cv2_load_flag)
            x2 = cv2.imread(os.path.join(self.cfg.data_dir, files[1]), self.cfg.cv2_load_flag)
            y = self.labels[i]
            return x1, x2, y
        except Exception as e:
            print(e)
            return None

    def __getitem__(self, i):
        data = self.get(i)
        while not isinstance(data, tuple):
            i = np.random.randint(len(self))
            data = self.get(i)

        x1, x2, y = data

        if self.cfg.channel_reverse and self.mode == "train" and bool(np.random.binomial(1, 0.5)):
            x1 = np.ascontiguousarray(x1[:, :, ::-1])
            x2 = np.ascontiguousarray(x2[:, :, ::-1])

        x1 = self.transforms(image=x1)["image"]
        x2 = self.transforms(image=x2)["image"]

        if x1.ndim == 2: x1 = np.expand_dims(x1, axis=-1)
        if x2.ndim == 2: x2 = np.expand_dims(x2, axis=-1)

        x1 = x1.transpose(2, 0, 1) # channels-last -> channels-first
        x1 = torch.tensor(x1).float()

        x2 = x2.transpose(2, 0, 1)
        x2 = torch.tensor(x2).float()

        y = torch.tensor(y).float()
        if y.ndim == 0:
            y = y.unsqueeze(-1)

        if self.cfg.convert_to_3d:
            x1 = x1.unsqueeze(0)
            x2 = x2.unsqueeze(0)

        return {"x1": x1, "x2": x2, "y": y, "index": i, "unique_id": self.unique_ids[i]}
