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
            df = df[df.fold != self.cfg.fold]
            self.transforms = self.cfg.train_transforms
        elif self.mode == "val":
            df = df[df.fold == self.cfg.fold]
            self.transforms = self.cfg.val_transforms

        self.dfs = [_df for series_id, _df in df.groupby("series_id")]
            
        self.collate_fn = train_collate_fn if mode == "train" else val_collate_fn


    def __len__(self):
        return len(self.dfs)
    
    @staticmethod
    def create_segmentation(series_df):
        seg = np.zeros((series_df.rows.values[0], series_df.cols.values[0], 5))
        r = int(0.02 * seg.shape[0])
        for row_idx, row in series_df.iterrows():
            tmp_seg = np.zeros_like(seg[..., 0])
            tmp_seg = cv2.circle(tmp_seg, (int(row.x), int(row.y)), r, 1, -1)
            seg[..., row_idx] = tmp_seg
        return seg

    def load_stack(self, i):
        series_df = self.dfs[i]
        series_df = series_df.reset_index(drop=True)
        study_id, series_id = str(series_df.study_id.values[0]), str(series_df.series_id.values[0])
        imgfiles = np.sort(glob.glob(os.path.join(self.cfg.data_dir, study_id, series_id, "*.png")))
        arr = np.stack([cv2.imread(_)[..., 1] for _ in imgfiles])
        arr = np.expand_dims(arr, axis=-1)
        instances = [os.path.basename(im).split("_")[-1].replace("INST", "").replace(".png", "") for im in imgfiles]
        instances = [int(_) for _ in instances]
        instance_to_position_index = {_: i for i, _ in enumerate(instances)}
        seg = self.create_segmentation(series_df)
        mask = np.zeros((arr.shape[0], arr.shape[1], arr.shape[2], 5))
        for row_idx, row in series_df.iterrows():
            mask[instance_to_position_index[row.instance_number], ..., row_idx] = seg[..., row_idx]
        # x_min, x_max = series_df.x_min.values[0], series_df.x_max.values[0]
        # x_min, x_max = max(0, x_min), min(x_max, arr.shape[2])
        # arr = arr[:, :, x_min:x_max]
        # mask = mask[:, :, x_min:x_max]
        if len(arr) < self.cfg.image_z:
            # pad 
            diff = self.cfg.image_z - len(arr)
            padding = np.expand_dims(np.zeros_like(arr[0]), axis=0)
            padding = np.repeat(padding, diff, axis=0)
            arr = np.concatenate([arr, padding])
            mask = np.concatenate([mask, np.repeat(padding, 5, axis=-1)])
        elif len(arr) > self.cfg.image_z:
            diff = len(arr) - self.cfg.image_z 
            diff = diff // 2
            arr = arr[diff:(diff + self.cfg.image_z)]
            mask = mask[diff:(diff + self.cfg.image_z)]

        assert len(arr) == len(mask) == self.cfg.image_z

        return arr, mask

    def __getitem__(self, i):
        x, y = self.load_stack(i)
        
        # if self.cfg.reverse_dim0 and self.mode == "train" and bool(np.random.binomial(1, 0.5)):
        #     x = np.ascontiguousarray(x[::-1])

        to_transform = {"image": x[0], "mask": y[0]}
        for idx in range(1, x.shape[0]):
            to_transform.update({f"image{idx}": x[idx], f"mask{idx}": y[idx]})

        transformed = self.transforms(**to_transform)
        x = np.stack([transformed["image"]] + [transformed[f"image{idx}"] for idx in range(1, x.shape[0])])
        x = torch.from_numpy(x)
        x = x.float()
        x = x.permute(3, 0, 1, 2)

        y = np.stack([transformed["mask"]] + [transformed[f"mask{idx}"] for idx in range(1, y.shape[0])])
        y = torch.from_numpy(y)
        y = y.float()
        y = y.permute(3, 0, 1, 2)

        return {"x": x, "y": y, "index": i}

