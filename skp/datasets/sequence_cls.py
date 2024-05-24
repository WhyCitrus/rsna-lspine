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

        self.inputs = df[self.cfg.inputs].tolist()
        self.labels = df[self.cfg.targets].values 

        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

        if "foldx" in self.cfg.data_dir:
            self.cfg.data_dir = self.cfg.data_dir.replace("foldx", f"fold{self.cfg.fold}")

    def __len__(self):
        return len(self.inputs) 

    def get(self, i):
        try:
            x = np.load(os.path.join(self.cfg.data_dir, self.inputs[i]), self.cfg.cv2_load_flag)
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
        # x.shape = (seq_len, dim_feat)
        if self.cfg.random_reverse_seq and self.mode == "train" and np.random.binomial(1, 0.5):
            x = np.ascontiguousarray(x[::-1])

        x = torch.tensor(x).float()

        if len(x) > self.cfg.seq_len:
            if self.cfg.resample_or_truncate == "resample":
                x = F.interpolate(x.unsqueeze(0).transpose(2, 1), self.cfg.seq_len).transpose(1, 2).squeeze(0)
                mask = torch.ones((x.shape[0], ))
            elif self.cfg.resample_or_truncate == "truncate":
                x = x[:self.cfg.seq_len]
                mask = torch.ones((x.shape[0], ))
        elif len(x) < self.cfg.seq_len:
            if self.cfg.resample_or_pad == "resample":
                x = F.interpolate(x.unsqueeze(0).transpose(2, 1), self.cfg.seq_len).transpose(1, 2).squeeze(0)
                mask = torch.ones((x.shape[0], ))
            elif self.cfg.resample_or_pad == "pad":
                fill = torch.zeros((self.cfg.seq_len - len(x), x.shape[1]))
                mask = torch.cat([torch.ones((x.shape[0], )), torch.zeros((fill.shape[0]), )])
                x = torch.cat([x, fill])
        else:
            mask = torch.ones((x.shape[0], ))
            
        assert len(x) == self.cfg.seq_len

        if y.ndim == 0:
            y = torch.tensor(y).float().unsqueeze(-1)

        return {"x": x, "y": y, "mask": mask, "index": i}
