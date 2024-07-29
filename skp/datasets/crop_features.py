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

        self.inputs = df[self.cfg.input].tolist()
        self.labels = df[self.cfg.targets].values

        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

        if "foldx" in self.cfg.data_dir:
            self.cfg.data_dir = self.cfg.data_dir.replace("foldx", f"fold{self.cfg.fold}")

    def __len__(self):
        return len(self.inputs) 

    def get(self, i):
        try:
            x = np.load(os.path.join(self.cfg.data_dir, self.inputs[i]))
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
        if self.mode == "train" and np.random.binomial(1, 0.5) and self.cfg.reverse_seq_aug:
            x = np.ascontiguousarray(x[::-1])

        if self.mode == "train" and np.random.binomial(1, 0.5) and self.cfg.shuffle_seq_aug:
            x = x[np.random.permutation(np.arange(len(x))).astype("int")]

        # should all be length 27 so no padding tokens are needed
        mask = torch.tensor([False] * len(x))

        x = torch.tensor(x).float()
        y = torch.tensor(y).float()

        return {"x": x, "y": y, "mask": mask, "index": i}
