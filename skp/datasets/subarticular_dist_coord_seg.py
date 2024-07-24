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
        self.inputs = df[self.cfg.inputs].tolist()
        self.labels = df[self.cfg.targets].values 

        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

        self.pixel_value_to_side_index = {175: 0, 255: 1}
        self.pixel_value_to_side_index_half = {int(k / 2): v for k, v in self.pixel_value_to_side_index.items()}

    def __len__(self):
        return len(self.inputs) 

    def get(self, i):
        try:
            assert self.cfg.cv2_load_flag == cv2.IMREAD_COLOR
            x = cv2.imread(os.path.join(self.cfg.data_dir, self.inputs[i]), self.cfg.cv2_load_flag)
            segfile = os.path.join(self.cfg.seg_dir, self.inputs[i])
            if not os.path.exists(segfile):
                # If no segmentation file exists, there is no segmentation (empty mask)
                seg_mask = np.zeros((x.shape[0], x.shape[1], 2)).astype("float32")
            else:
                segfile = cv2.imread(segfile, 0)
                # segfile is saved as grayscale image
                # make a 2-channel mask 
                seg_mask = np.zeros((x.shape[0], x.shape[1], 2)).astype("float32")
                for k, v in self.pixel_value_to_side_index_half.items():
                    seg_mask[:, :, v][segfile == k] = 1
                for k, v in self.pixel_value_to_side_index_half.items():
                    seg_mask[:, :, v][segfile == k] = 0.5    
            if x.shape[:2] != seg_mask.shape[:2]:
                seg_mask = cv2.resize(seg_mask, x.shape[:2])
            assert x.shape[:2] == seg_mask.shape[:2]
            y = self.labels[i]
            return x, y, seg_mask
        except Exception as e:
            print(e)
            return None

    def __getitem__(self, i):
        data = self.get(i)
        while not isinstance(data, tuple):
            i = np.random.randint(len(self))
            data = self.get(i)

        x, y, mask = data
        transformed = self.transforms(image=x, mask=mask)
        x, mask = transformed["image"], transformed["mask"]
        x, mask = x.transpose(2, 0, 1), mask.transpose(2, 0, 1)
        x, mask = torch.from_numpy(x).float(), torch.from_numpy(mask).float()
        y = torch.tensor(y).float()

        return {"x": x, "y_cls": y, "y_seg": mask, "index": i}
