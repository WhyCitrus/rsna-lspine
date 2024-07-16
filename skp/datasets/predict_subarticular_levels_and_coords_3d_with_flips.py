import copy
import cv2
import glob
import numpy as np
import os
import pandas as pd
import pickle
import torch

from scipy.ndimage import zoom
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

    @staticmethod
    def get_indices_for_levels_stack_crop(level_ranges, levels_to_include):
        # levels must be consecutive
        indices = []
        for level in levels_to_include:
           indices.extend(list(level_ranges[level]))
        return (min(indices), max(indices))

    def crop_to_level_range(self, arr, i):
        # we will turn this on even for validation
        # since otherwise all of the series will have all the levels
        # will introduce some stochasticity into val process but it's ok
        level_ranges = self.annotations[i]["level_ranges"]

        levels = [*level_ranges]
        if np.random.binomial(1, 0.5): 
            # Half of the time crop levels
            num_levels_to_crop = np.random.choice([4, 3, 2, 1], 1, p=[0.45, 0.3, 0.15, 0.1], replace=False)[0]
            levels_to_include = np.random.choice(np.arange(5 - num_levels_to_crop + 1), 1, replace=False)[0]
            levels_to_include = levels[levels_to_include:(levels_to_include + num_levels_to_crop)]
            indices = self.get_indices_for_levels_stack_crop(level_ranges, levels_to_include) 
            arr = arr[indices[0]:indices[1] + 1]
            level_labels = np.asarray([1 if each_level in levels_to_include else 0 for each_level in levels])
            # Need to edit slice index annotation if cropping
            ann = copy.deepcopy(self.annotations[i])
            for each_level in levels_to_include:
                # left
                slice_coord = ann["coords"][f"L_{each_level}"][0]
                new_slice_coord = (slice_coord - indices[0]) / (len(arr) - 1)
                ann["coords"][f"L_{each_level}"][0] = new_slice_coord
                # right
                slice_coord = ann["coords"][f"R_{each_level}"][0]
                new_slice_coord = (slice_coord - indices[0]) / (len(arr) - 1)
                ann["coords"][f"R_{each_level}"][0] = new_slice_coord

        else:
            ann = copy.deepcopy(self.annotations[i])
            level_labels = np.ones((5, ))
            for each_level in levels:
                # left
                ann["coords"][f"L_{each_level}"][0] = ann["coords"][f"L_{each_level}"][0] / (ann["num_slices"] - 1)
                # right
                ann["coords"][f"R_{each_level}"][0] = ann["coords"][f"R_{each_level}"][0] / (ann["num_slices"] - 1)

        coords = []
        for each_level in levels:
            coords.extend(ann["coords"][f"R_{each_level}"])

        for each_level in levels:
            coords.extend(ann["coords"][f"L_{each_level}"])

        if self.cfg.image_z != len(arr):
            indices = np.arange(len(arr))
            indices = zoom(indices, self.cfg.image_z / len(arr), order=0, prefilter=False).astype("int")
            assert len(indices) == self.cfg.image_z
            arr = arr[indices]

        return arr, np.asarray(coords), level_labels

    def load_stack_and_annotation(self, i):
        # assumes that inputs are list of directories
        # where each directory contains all the images in a stack as PNG 
        # and that filenames are sortable
        images = np.sort(glob.glob(os.path.join(self.cfg.data_dir, self.annotations[i]["series_folder"], "*.png")))
        x = np.stack([cv2.imread(im, self.cfg.cv2_load_flag) for im in images], axis=0)
        x, coords, level_labels = self.crop_to_level_range(x, i)
        if x.ndim == 3:
            x = np.expand_dims(x, axis=-1)
        # x.shape = (Z, H, W, C)
        return x, coords, level_labels

    def get(self, i):
        try:
            x, coords, level_labels = self.load_stack_and_annotation(i)
            return x, coords, level_labels
        except Exception as e:
            print("ERROR:", self.annotations[i]["series_folder"], "\n", e)
            return None

    def __getitem__(self, i):
        data = self.get(i)
        while not isinstance(data, tuple):
            i = np.random.randint(len(self))
            data = self.get(i)

        x, coords, level_labels = data
        included_levels = np.concatenate([[_] * 3 for _ in level_labels])

        if self.mode == "train" and np.random.binomial(1, 0.5):
            # flip along slice dimension
            x = np.ascontiguousarray(x[::-1])
            # edit slice coordinates
            coords[[0, 3, 6, 9, 12]] = 1 - coords[[0, 3, 6, 9, 12]] 
            coords[[15, 18, 21, 24, 27]] = 1 - coords[[15, 18, 21, 24, 27]]

        if self.mode == "train" and np.random.binomial(1, 0.5):
            # flip up-down
            x = np.ascontiguousarray(x[:, ::-1])
            # edit y coordinates
            coords[[2, 5, 8, 11, 14]] = 1 - coords[[2, 5, 8, 11, 14]]
            coords[[17, 20, 23, 26, 29]] = 1 - coords[[17, 20, 23, 26, 29]]

        if self.mode == "train" and np.random.binomial(1, 0.5):
            # flip left-right
            # edit slice coordinates
            # left becomes 1 - right
            # and right becomes 1 - left
            # first 15 elements are right, last 15 are left
            x = np.ascontiguousarray(x[:, :, ::-1])
            # edit x coordinates
            coords[[1, 4, 7, 10, 13]] = 1 - coords[[16, 29, 22, 25, 28]]
            coords[[16, 19, 22, 25, 28]] = 1 - coords[[1, 4, 7, 10, 13]]

        to_transform = {"image": x[0]}
        for idx in range(1, x.shape[0]):
            to_transform.update({f"image{idx}": x[idx]})

        xt = self.transforms(**to_transform)
        x = np.stack([xt["image"]] + [xt[f"image{idx}"] for idx in range(1, x.shape[0])])
        x = torch.from_numpy(x)
        x = x.float()
        x = x.permute(3, 0, 1, 2)

        coords = torch.tensor(coords).float()
        level_labels = torch.tensor(level_labels).float()
        included_levels = torch.tensor(included_levels).long()

        return {"x": x, "coords": coords, "level_labels": level_labels, "included_levels": included_levels, "index": i}
