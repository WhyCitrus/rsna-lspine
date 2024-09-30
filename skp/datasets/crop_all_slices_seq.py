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
            if "ignore_during_val" in df.columns:
                df = df[df.ignore_during_val == 0]
            self.transforms = self.cfg.val_transforms

        df["crop_index"] = df.filepath.apply(lambda x: x.split("_")[-1].replace(".png", ""))
        df["study_level_laterality_crop"] = df.study_level_laterality + "_" + df.crop_index
        self.df = [_df for _, _df in df.groupby("study_level_laterality_crop")]

        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn

    def __len__(self):
        return len(self.df) 

    def get(self, i):
        try:
            df = self.df[i]
            df = df.sort_values("position_index")
            if len(df) > self.cfg.max_num_images:
                # Trim off both ends equally if exceeds number of images
                diff = len(df) - self.cfg.max_num_images
                diff = diff // 2
                df = df.iloc[diff:(diff + self.cfg.max_num_images)]
            x = [cv2.imread(os.path.join(self.cfg.data_dir, fp), self.cfg.cv2_load_flag) for fp in df.filepath.tolist()]
            y = df[self.cfg.targets].values
            return x, y, df.unique_id.values[0]
        except Exception as e:
            print(e)
            return None

    def __getitem__(self, i):
        data = self.get(i)
        while not isinstance(data, tuple):
            i = np.random.randint(len(self))
            data = self.get(i)

        x, y, unique_id = data

        to_transform = {"image": x[0]}
        for idx in range(1, len(x)):
            to_transform.update({f"image{idx}": x[idx]})

        xt = self.transforms(**to_transform)
        x = np.stack([xt["image"]] + [xt[f"image{idx}"] for idx in range(1, len(x))])

        if x.ndim == 3:
            x = np.expand_dims(x, axis=-1)

        mask = [False] * len(x)

        prepad_length = len(x)

        if len(x) < self.cfg.max_num_images:
            diff = self.cfg.max_num_images - len(x)
            xpad = np.expand_dims(np.zeros_like(x[0]), axis=0)
            x = np.concatenate([x] + [xpad] * diff)
            ypad = np.expand_dims(np.zeros_like(y[0]), axis=0)
            y = np.concatenate([y] + [ypad] * diff)
            mask = mask + [True] * diff

        x = x.transpose(0, 3, 1, 2) # channels-last -> channels-first
        x = torch.tensor(x).float()
        y = torch.tensor(y).float()

        if self.cfg.convert_to_3d:
            x = x.unsqueeze(1)

        if self.mode == "val":
            return {"x": x, "y": y, "index": i, "unique_id": torch.tensor([unique_id] * len(x)), "mask": torch.tensor(mask)}
        else:
            return {"x": x, "y": y, "index": i, "mask": torch.tensor(mask)}
