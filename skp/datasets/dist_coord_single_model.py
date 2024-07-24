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
            annotations = pickle.load(f)
        if self.mode == "train":
            self.annotations = [_ for _ in annotations if _["fold"] != self.cfg.fold]
            self.transforms = self.cfg.train_transforms
        elif self.mode == "val":
            self.annotations = [_ for _ in annotations if _["fold"] == self.cfg.fold]
            self.transforms = self.cfg.val_transforms

        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self):
        return len(self.annotations) 

    def get(self, i):
        try:
            assert self.cfg.cv2_load_flag == cv2.IMREAD_GRAYSCALE
            filepaths = self.annotations[i]["filepaths"]
            # should already be sorted
            x = np.expand_dims(np.stack([cv2.imread(os.path.join(self.cfg.data_dir, fp), self.cfg.cv2_load_flag) for fp in filepaths]), axis=-1)
            y_dist, y_coord, positions = self.annotations[i]["dist_labels"], self.annotations[i]["coord_labels"], self.annotations[i]["positions"]
            return x, y_dist, y_coord, positions
        except Exception as e:
            print(e)
            return None

    def __getitem__(self, i):
        data = self.get(i)
        while not isinstance(data, tuple):
            i = np.random.randint(len(self))
            data = self.get(i)

        x, y_dist, y_coord, positions = data

        to_transform = {"image": x[0]}
        for idx in range(1, x.shape[0]):
            to_transform.update({f"image{idx}": x[idx]})

        xt = self.transforms(**to_transform)
        x = np.stack([xt["image"]] + [xt[f"image{idx}"] for idx in range(1, x.shape[0])])

        num_images = self.cfg.num_images
        if len(x) > num_images:
            x = x[:num_images]
            y_dist = y_dist[:num_images]
            positions = positions[:num_images]
            padding = [False] * num_images
        elif len(x) < num_images:
            diff = num_images - len(x)
            padding = [False] * len(x) + [True] * diff
            img_pad = np.zeros((diff, x.shape[1], x.shape[2], x.shape[3]))
            x = np.concatenate([x, img_pad])
            dist_pad = np.zeros((diff, y_dist.shape[1]))
            y_dist = np.concatenate([y_dist, dist_pad])
            pos_pad = np.zeros((diff, ))
            positions = np.concatenate([positions, pos_pad])
        else:
            padding = [False] * num_images

        x = torch.from_numpy(x)
        x = x.float()
        x = x.permute(0, 3, 1, 2) # (N, C, H, W)
        y_dist = torch.tensor(y_dist).float()
        y_coord = torch.tensor(y_coord).float()
        positions = torch.tensor(positions).float().unsqueeze(1)
        padding = torch.tensor(padding)

        return {"x": x, "y_dist": y_dist, "y_coord": y_coord, "positions": positions, "mask": padding, "index": i}
