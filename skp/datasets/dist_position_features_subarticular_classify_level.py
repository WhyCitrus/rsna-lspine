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
        col_names = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
        col_names = [_ + "_position_index" for _ in col_names]
        self.labels = df[col_names].values

        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

        if "foldx" in self.cfg.data_dir:
            self.cfg.data_dir = self.cfg.data_dir.replace("foldx", f"fold{self.cfg.fold}")

    def __len__(self):
        return len(self.inputs) 

    @staticmethod
    def sample_level(level_positions, x):
        # level_positions.shape = (5, ) --> l1_l2, l2_l3, l3_l4, l4_l5, l5_s1
        # randomly select a percentage of the original sequence to sample
        if np.random.binomial(1, 0.4):
            sample_num = len(x)
            start_index = 0
        else:
            sample_pct = np.random.uniform(0.25, 1.0)
            sample_num = int(sample_pct * len(x))
            # randomly sample a start index
            start_index = int(np.random.uniform(0, len(x) - sample_num + 1))
        level_labels = [0, 0, 0, 0, 0]
        included_indices = np.arange(start_index, start_index + sample_num)
        assert len(included_indices) == sample_num, f"len(included_indices) is {len(included_indices)} while sample_num is {sample_num}"
        for idx, level_pos in enumerate(level_positions):
            if level_pos in included_indices:
                level_labels[idx] = 1
        x_copy = x[start_index:start_index+sample_num].copy()
        assert len(x_copy) == sample_num, f"len(x) is {len(x_copy)} while sample_num is {sample_num}"
        return x_copy, np.asarray(level_labels)

    def get(self, i):
        try:
            x = np.load(os.path.join(self.cfg.data_dir, self.inputs[i]))
            x, y = self.sample_level(self.labels[i], x)
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
                x = x[:max_len]
                mask = torch.tensor([False] * max_len) # True indicates padding token
            elif x.shape[0] < max_len:
                diff = max_len - x.shape[0]
                orig_len = x.shape[0]
                x_pad = np.zeros((diff, x.shape[1]))
                x = np.concatenate([x, x_pad])
                mask = torch.concatenate([torch.tensor([False] * orig_len), torch.tensor([True] * diff)])
                assert len(mask) == len(x)
            else:
                mask = torch.tensor([False] * max_len)
        else:
            # if val, we use batch size 1 so no need to pad
            mask = torch.tensor([False] * len(x))

        x = torch.tensor(x).float()
        y = torch.tensor(y).float()

        return {"x": x, "y": y, "mask": mask, "index": i}
