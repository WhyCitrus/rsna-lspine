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
            if len(x) > self.cfg.max_seq_len: 
                if len(x) <= self.cfg.max_seq_len * 2:
                    x = np.ascontiguousarray(x[::2])
                    y = np.ascontiguousarray(y[::2])
                elif len(x) <= self.cfg.max_seq_len * 3:
                    x = np.ascontiguousarray(x[::3])
                    y = np.ascontiguousarray(y[::3])
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

        if self.mode == "train":
            max_len = self.cfg.max_seq_len
            if x.shape[0] > max_len:
                x, y = x[:max_len], y[:max_len]
                mask = torch.tensor([False] * max_len) # True indicates padding token
            elif x.shape[0] < max_len:
                diff = max_len - x.shape[0]
                orig_len = x.shape[0]
                x_pad = np.zeros((diff, x.shape[1]))
                y_pad = np.zeros((diff, y.shape[1]))
                x, y = np.concatenate([x, x_pad]), np.concatenate([y, y_pad])
                mask = torch.concatenate([torch.tensor([False] * orig_len), torch.tensor([True] * diff)])
                assert len(mask) == len(x) == len(y), f"len(mask) is {len(mask)}, len(x) is {len(x)}, len(y) is {len(y)}"
            else:
                mask = torch.tensor([False] * max_len)
        else:
            # if val, we use batch size 1 so no need to pad
            mask = torch.tensor([False] * len(x))

        x = torch.tensor(x).float()
        y = torch.tensor(y).float()

        return {"x": x, "y": y, "mask": mask, "index": i}
