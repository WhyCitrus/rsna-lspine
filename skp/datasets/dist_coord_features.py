import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F

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

        self.inputs = df.features.tolist()
        self.labels = df.labels.tolist() 

        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

        if "foldx" in self.cfg.data_dir:
            self.cfg.data_dir = self.cfg.data_dir.replace("foldx", f"fold{self.cfg.fold}")

    def __len__(self):
        return len(self.inputs) 

    def get(self, i):
        try:
            x = np.load(os.path.join(self.cfg.data_dir, self.inputs[i]))
            y = np.load(os.path.join(self.cfg.data_dir, self.labels[i]))
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
        # x.shape = (seq_len, dim_feat)
        # y.shape = (seq_len, 30) --> first 10 are right and left dist, last 20 are foramen coordinates
        y_dist = y[:, :10]
        y_coord = y[0, 10:]

        if self.mode == "train":
            max_len = self.cfg.max_seq_len
            if x.shape[0] > max_len:
                x, y_dist = x[:max_len], y_dist[:max_len]
                mask = torch.tensor([False] * max_len) # True indicates padding token
            elif x.shape[0] < max_len:
                diff = max_len - x.shape[0]
                orig_len = x.shape[0]
                x_pad = np.zeros((diff, x.shape[1]))
                y_pad = np.zeros((diff, y_dist.shape[1]))
                x, y_dist = np.concatenate([x, x_pad]), np.concatenate([y_dist, y_pad])
                mask = torch.concatenate([torch.tensor([False] * orig_len), torch.tensor([True] * diff)])
                assert len(mask) == len(x) == len(y_dist)
            else:
                mask = torch.tensor([False] * max_len)
        else:
            # if val, we use batch size 1 so no need to pad
            mask = torch.tensor([False] * len(x))

        x = torch.tensor(x).float()
        y_dist = torch.tensor(y_dist).float()
        y_coord = torch.tensor(y_coord).float()

        return {"x": x, "y_dist": y_dist, "y_coord": y_coord, "mask": mask, "index": i}
