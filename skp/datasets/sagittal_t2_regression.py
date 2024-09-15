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
            self.transforms = self.cfg.val_transforms

        self.df = df.reset_index(drop=True)
        self.inputs = df[cfg.inputs].values
        self.labels = df[cfg.targets].values
        if "sampling_weight" in df.columns:
            self.sampling_weights = df.sampling_weight.values
            
        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self):
        return len(self.inputs)
    
    def get(self, i):
        x = cv2.imread(os.path.join(self.cfg.data_dir, self.inputs[i]))
        y = self.labels[i]
        return x, y

    def __getitem__(self, i):
        data = self.get(i)
        while not isinstance(data, tuple):
            i = np.random.randint(len(self))
            data = self.get(i)

        x, y = data 
        # turn y into keypoint labels
        # y.shape = 10; first 5 is x, second 5 is y
        keypoints = [(yi, yj) for yi, yj in zip(y[:5], y[5:])]
        transformed = self.transforms(image=x, keypoints=keypoints)
        x, keypoints = transformed["image"], transformed["keypoints"]
        y = np.asarray([kp[0] for kp in keypoints] + [kp[1] for kp in keypoints])
        y[:5] = y[:5] / x.shape[1]
        y[5:] = y[5:] / x.shape[0]
        x = x.transpose(2, 0, 1)
        x, y = torch.from_numpy(x), torch.from_numpy(y)

        return {"x": x, "y": y, "index": i}
