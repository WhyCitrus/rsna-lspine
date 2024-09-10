import cv2
import glob
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
            if not self.cfg.fullfit:
                df = df[df.fold != self.cfg.fold]
            self.transforms = self.cfg.train_transforms
        elif self.mode == "val":
            df = df[df.fold == self.cfg.fold]
            self.transforms = self.cfg.val_transforms

        self.series_folder = df.series_folder.tolist()
        self.df = df
            
        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn


    def __len__(self):
        return len(self.df)

    def load_stack(self, i):
        imgfiles = np.sort(glob.glob(os.path.join(self.cfg.data_dir, self.series_folder[i], "IM*")))
        maskfiles = [_.replace("IM", "MASK") for _ in imgfiles]
        arr = np.stack([cv2.imread(_, 0) for _ in imgfiles])
        mask = np.stack([cv2.imread(_, 0) for _ in maskfiles])
        if len(arr) < self.cfg.image_z:
            # pad 
            diff = self.cfg.image_z - len(arr)
            padding = np.expand_dims(np.zeros_like(arr[0]), axis=0)
            padding = np.repeat(padding, diff, axis=0)
            arr = np.concatenate([arr, padding])
            mask = np.concatenate([mask, padding])
        elif len(arr) > self.cfg.image_z:
            diff = len(arr) - self.cfg.image_z 
            diff = diff // 2
            arr = arr[diff:(diff + self.cfg.image_z)]
            mask = mask[diff:(diff + self.cfg.image_z)]

        assert len(arr) == len(mask) == self.cfg.image_z

        return arr, mask

    def __getitem__(self, i):
        x, y = self.load_stack(i)
        # x.shape = y.shape = (N, H, W)

        # if self.cfg.reverse_dim0 and self.mode == "train" and bool(np.random.binomial(1, 0.5)):
        #     x = np.ascontiguousarray(x[::-1])

        to_transform = {"image": x[0], "mask": y[0]}
        for idx in range(1, x.shape[0]):
            to_transform.update({f"image{idx}": x[idx], f"mask{idx}": y[idx]})

        transformed = self.transforms(**to_transform)
        x = np.stack([transformed["image"]] + [transformed[f"image{idx}"] for idx in range(1, x.shape[0])])
        x = torch.from_numpy(x)
        x = x.float()
        x = x.unsqueeze(0)

        y = np.stack([transformed["mask"]] + [transformed[f"mask{idx}"] for idx in range(1, y.shape[0])])
        y = torch.from_numpy(y)
        # convert to one_hot
        y = torch.nn.functional.one_hot(y.long(), num_classes=20)[..., 1:]
        y = y.float()
        y = y.permute(3, 0, 1, 2)

        return {"x": x, "y": y, "index": i}

