import cv2
import numpy as np
import os
import pickle
import torch

from torch.utils.data import Dataset as TorchDataset


def collate_fn(batch):
    images = [torch.tensor(sample["img"]).float() for sample in batch]
    targets = [{"boxes": torch.tensor(sample["boxes"]).float(), "labels": torch.tensor(sample["labels"], dtype=torch.int64)} for sample in batch]
    return {"images": images, "targets": targets}


train_collate_fn = collate_fn
val_collate_fn = collate_fn


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
        
    def __getitem__(self, i):
        ann = self.annotations[i]
        img = cv2.imread(os.path.join(self.data_dir, ann["filepath"]), self.cfg.cv2_load_flag)
        transformed = self.transforms(image=img, bboxes=ann["bboxes"], class_labels=ann["labels"])
        img, bboxes, labels = transformed["image"], np.asarray(transformed["bboxes"]), np.asarray(transformed["class_labels"])
        img = img.transpose(2, 0, 1)
        return {"img": img / 255., "boxes": bboxes, "labels": labels}