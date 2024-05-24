import cv2
import glob
import numpy as np
import os
import pandas as pd
import torch

from collections import defaultdict
from torch.nn.functional import one_hot
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
        self.labels = df[self.cfg.targets].tolist() 

        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

        if isinstance(self.cfg.subset_segmentations, list):
            map_dict = {v: idx + 1 for idx, v in enumerate(self.cfg.subset_segmentations)}
            map_dict[0] = 0 
            self.map_func = np.vectorize(map_dict.get)

    def __len__(self):
        return len(self.inputs) 

    def load_images_from_folder(self, folder, cv2_flag):
        image_list = np.sort(glob.glob(os.path.join(self.cfg.data_dir, folder, "*")))
        images = np.stack([cv2.imread(im, cv2_flag) for im in image_list])
        return images 

    def get(self, i):
        x = self.load_images_from_folder(self.inputs[i], cv2_flag=cv2.IMREAD_COLOR).astype("float")
        y = self.load_images_from_folder(self.labels[i], cv2_flag=cv2.IMREAD_GRAYSCALE).astype("int")
        assert len(x) == len(y)

        if self.mode == "train":
            if isinstance(self.cfg.randomly_drop_axial_slices, float):
                if np.random.binomial(1, self.cfg.randomly_drop_axial_slices):
                    # TotalSegmentator dataset is resampled isotropically to 1.5 mm
                    # In practice, axial slices may be 3 or 5 mm slices 
                    # This augmentation will drop out either every other slice (3 mm)
                    # or 2 out of every 3 slices (4.5 mm)
                    if np.random.binomial(1, 0.5):
                        # Drop every other slice
                        x, y = x[::2], y[::2]
                    else:
                        # Drop 2 out of every 3 slices
                        x, y = x[::3], y[::3]

        # need to move channels first for monai transforms
        x = np.moveaxis(x, -1, 0)
        if isinstance(self.cfg.subset_segmentations, list):
            y[~np.isin(y, self.cfg.subset_segmentations)] = 0
            y = self.map_func(y)
        y = np.expand_dims(y, axis=0)
        return {"image": x, "label": y}

    def get_wrapper(self, i):
        if self.cfg.dataset_ignore_errors:
            try:
                return self.get(i)
            except Exception as e:
                print(e)
                return None 
        else:
            # when debugging should not ignore errors
            # when training for long time, can be useful to ignore
            # infrequent errors due to corrupted data, etc. so it 
            # doesn't crash due to 1 sample
            return self.get(i)

    def __getitem__(self, i):
        data = self.get_wrapper(i)
        while not isinstance(data, dict):
            i = np.random.randint(len(self))
            data = self.get_wrapper(i)

        data = self.transforms(data)
        data["label"] = data["label"].squeeze(0)
        data["label"] = one_hot(data["label"].long(), num_classes=self.cfg.num_classes + 1)[..., 1:].movedim(-1, 0)

        return {"x": data["image"].float(), "y": data["label"], "index": i}
