import cv2
import glob
import nibabel as nib
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
            df = df[df.fold == self.cfg.fold]
            self.transforms = self.cfg.val_transforms

        self.inputs = df[self.cfg.inputs].tolist()
        self.labels = df[self.cfg.targets[0]].tolist() 

        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self):
        return len(self.inputs) 

    def load_4channels(self, i):
        # 1- FLAIR, 2- T1c, 3- T1, 4- T2
        x = np.stack([
                nib.load(os.path.join(self.inputs[i], "FLAIR.nii.gz")).get_fdata(),
                nib.load(os.path.join(self.inputs[i], "SPGR.nii.gz")).get_fdata(),
                nib.load(os.path.join(self.inputs[i], "T1.nii.gz")).get_fdata(),
                nib.load(os.path.join(self.inputs[i], "T2.nii.gz")).get_fdata()
            ])
        return x

    def load_labels(self, i):
        y = np.stack([
                nib.load(os.path.join(self.labels[i], "Combo_label.nii.gz")).get_fdata(),
                nib.load(os.path.join(self.labels[i], "SPGR_label.nii.gz")).get_fdata()
            ])
        return y 

    def get(self, i):
        try:
            x = self.load_4channels(i)
            y = self.load_labels(i)
            return {"image": x, "label": y}
        except Exception as e:
            print(e)
            return None

    def __getitem__(self, i):
        data = self.get(i)
        while not isinstance(data, dict):
            i = np.random.randint(len(self))
            data = self.get(i)

        data = self.transforms(data)

        return {"x": data["image"], "y": data["label"], "index": i}
