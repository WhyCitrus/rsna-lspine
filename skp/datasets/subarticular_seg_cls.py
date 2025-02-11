import cv2
import numpy as np
import os
import pickle
import torch

from torch.utils.data import Dataset as TorchDataset, default_collate


train_collate_fn = default_collate
val_collate_fn = default_collate


class Dataset(TorchDataset):

    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode
        with open(cfg.annotations_file, "rb") as f:
            annotations = pickle.load(f)
        self.data_dir = cfg.data_dir
        if self.mode == "train":
            self.annotations = [_ for _ in annotations if _["fold"] != self.cfg.fold]
            self.transforms = self.cfg.train_transforms
        elif self.mode == "val":
            self.annotations = [_ for _ in annotations if _["fold"] == self.cfg.fold]
            self.transforms = self.cfg.val_transforms
        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self):
        return len(self.annotations)
    
    @staticmethod
    def create_mask(ann, shape):
        rt_mask = np.zeros((shape[0], shape[1]))
        lt_mask = np.zeros_like(rt_mask)
        if len(ann["rt_coords"]) > 0:
            rt_mask = cv2.ellipse(rt_mask, (int(ann["rt_coords"][0]), int(ann["rt_coords"][1])), (int(0.025*shape[0]), int(0.025*shape[1])), 0, 0, 360, 1, -1)
        if len(ann["lt_coords"]) > 0:
            lt_mask = cv2.ellipse(lt_mask, (int(ann["lt_coords"][0]), int(ann["lt_coords"][1])), (int(0.025*shape[0]), int(0.025*shape[1])), 0, 0, 360, 1, -1)
        mask = np.stack([rt_mask, lt_mask], axis=-1)
        return mask

    def __getitem__(self, i):
        ann = self.annotations[i]
        img = cv2.imread(os.path.join(self.data_dir, ann["filepath"]), self.cfg.cv2_load_flag)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        mask = self.create_mask(ann, img.shape[:2])
        transformed = self.transforms(image=img, mask=mask)
        img, mask = transformed["image"], transformed["mask"]
        img, mask = img.transpose(2, 0, 1), mask.transpose(2, 0, 1)
        img, mask = torch.from_numpy(img), torch.from_numpy(mask)
        labels = ann["labels"]
        return {"x": img, "y_seg": mask, "y_cls": labels}
