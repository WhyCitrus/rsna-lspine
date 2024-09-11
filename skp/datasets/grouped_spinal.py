import cv2
import numpy as np
import os
import pandas as pd
import pickle
import torch

from torch.utils.data import Dataset as TorchDataset, default_collate


train_collate_fn = default_collate
val_collate_fn = default_collate


class Dataset(TorchDataset):

    def __init__(self, cfg, mode):
        self.cfg = cfg 
        self.mode = mode
        with open(self.cfg.annotations_file, "rb") as f:
            ann = pickle.load(f)
        if self.mode == "train":
            ann = [a for a in ann if a["fold"] != cfg.fold]
            self.transforms = self.cfg.train_transforms
        elif self.mode == "val":
            ann = [a for a in ann if a["fold"] == cfg.fold]
            self.transforms = self.cfg.val_transforms

        self.ann = ann
        self.levels = ["L1_L2", "L2_L3", "L3_L4", "L4_L5", "L5_S1"]
        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self):
        return len(self.ann) 

    def get(self, i):
        try:
            image_list = []
            for each_level in self.levels:
                level_files = self.ann[i]["files"][each_level]
                image_list.append(np.random.choice(level_files))
            x = [cv2.imread(os.path.join(self.cfg.data_dir, img), self.cfg.cv2_load_flag) for img in image_list]
            y = self.ann[i]["labels"]
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

        to_transform = {"image": x[0]}
        for idx in range(1, len(x)):
            to_transform.update({f"image{idx}": x[idx]})

        xt = self.transforms(**to_transform)
        x = np.stack([xt["image"]] + [xt[f"image{idx}"] for idx in range(1, len(x))])
        x = torch.from_numpy(x)
        x = x.float()
        # x.shape = (5, H, W, C)

        x = x.permute(0, 3, 1, 2) # channels-last -> channels-first
        y = torch.from_numpy(y)
        # y.shape = (5, 3)

        return {"x": x, "y": y, "index": i}
