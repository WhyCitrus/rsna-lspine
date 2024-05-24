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
        prefix = os.path.basename(self.inputs[i])
        x = np.stack([
                nib.load(os.path.join(self.inputs[i], f"{prefix}-t2f.nii.gz")).get_fdata(),
                nib.load(os.path.join(self.inputs[i], f"{prefix}-t1c.nii.gz")).get_fdata(),
                nib.load(os.path.join(self.inputs[i], f"{prefix}-t1n.nii.gz")).get_fdata(),
                nib.load(os.path.join(self.inputs[i], f"{prefix}-t2w.nii.gz")).get_fdata()
            ])
        return x

    def load_labels(self, i):
        prefix = os.path.basename(self.labels[i])
        orig = np.expand_dims(nib.load(os.path.join(self.labels[i], f"{prefix}-seg.nii.gz")).get_fdata(), axis=0)
        # original:
        # 0- background
        # 1- necrotic
        # 2- peritumoral edematous/invaded tissue
        # 3- enhancing tumor
        #
        # BraTS subregions:
        # enhancing tumor = 3
        # tumor core = 3 + 1
        # whole tumor = 3 + 2 + 1
        y = np.zeros_like(orig)
        y = np.repeat(y, 3, axis=0)
        if self.cfg.convert_labels_to_brats_subregions:
            y[0] = (orig > 0).astype("float")
            y[1] = y[0] - (orig == 2).astype("float")
            y[2] = (orig == 3).astype("float")
        else:
            y[0] = (orig == 2).astype("float")
            y[1] = (orig == 3).astype("float")
            y[2] = (orig == 1).astype("float")
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
