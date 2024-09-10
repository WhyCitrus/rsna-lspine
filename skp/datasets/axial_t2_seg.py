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
        self.files = df.filepath.values
        if "sampling_weight" in df.columns:
            self.sampling_weights = df.sampling_weight.values
            
        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self):
        return len(self.files)
    
    def get(self, i):
        img = cv2.imread(os.path.join(self.cfg.data_dir, self.files[i]))
        seg = cv2.imread(os.path.join(self.cfg.seg_data_dir, self.files[i]))
        seg = seg.astype("float") / 255.0
        if img.shape != seg.shape:
            print(f"ERROR {self.files[i]} : image shape {img.shape} does not match mask shape {seg.shape}")
            return None
        return (img, seg)

    def __getitem__(self, i):
        data = self.get(i)
        while not isinstance(data, tuple):
            i = np.random.randint(len(self))
            data = self.get(i)

        img, seg = data
        transformed = self.transforms(image=img, mask=seg)
        img, seg = transformed["image"], transformed["mask"]
        img, seg = img.transpose(2, 0, 1), seg.transpose(2, 0, 1)
        img, seg = torch.from_numpy(img), torch.from_numpy(seg)

        return {"x": img, "y": seg, "index": i}
