"""
TotalSegmentator dataset downloaded from:
https://zenodo.org/records/10047292, Mar 16, 2024
v2.0.1

Convert from original NIfTI to PNGs for ease of training.

Use 3 windows (from https://radiopaedia.org/articles/windowing-ct):
- Brain: WL=40, WW=80
- Mediastinum: WL=50, WW=350
- Bone: WL=400, WW=1800

Also combine all of the individual class segmentation NIfTI files 
into 1.

Each class should be mutually exclusive so this should be OK. There
may be instances where this is not true in the data due to noise in
labels at the edge of adjacent structures, but we will ignore this
for simplicity's sake.
"""

import cv2
import glob
import nibabel as nib
import numpy as np
import os

from tqdm import tqdm


def window(x, WL, WW):
    upper, lower = WL + WW // 2, WL - WW // 2
    x = np.clip(x, lower, upper)
    x = (x - lower) / (upper - lower)
    x = (x * 255).astype("uint8")
    return x


def load_and_reorient_nifti(nii):
    nii = nib.load(nii).get_fdata()
    # assumes 3D image
    assert nii.ndim == 3
    nii = np.rot90(nii, axes=(0, 1))
    nii = np.flip(nii, axis=1)
    nii = nii.transpose(2, 0, 1)[::-1]
    # 1st dimension is slice dimension
    # 2nd dimension is AP (assuming axial orientation)
    # 3rd dimension is RL
    return nii


save_dir = "/mnt/stor/datasets/totalsegmentator/pngs-v201/"
os.makedirs(save_dir, exist_ok=True)

studies = glob.glob("/mnt/stor/datasets/totalsegmentator/v201/*")
studies = [_ for _ in studies if os.path.isdir(_)]

cts = [os.path.join(st, "ct.nii.gz") for st in studies]

for each_ct in tqdm(cts):
    study_id = os.path.basename(os.path.dirname(each_ct))
    tmp_ct = load_and_reorient_nifti(each_ct)
    # negative HU should exist on every scan...
    assert np.min(tmp_ct) < 0
    tmp_ct = np.stack([
            window(tmp_ct, WL=40, WW=80),
            window(tmp_ct, WL=50, WW=350),
            window(tmp_ct, WL=400, WW=1800)
        ], axis=-1)
    tmp_ct_save_dir = os.path.join(save_dir, study_id, "images")
    os.makedirs(tmp_ct_save_dir, exist_ok=True)
    for idx, each_ct_slice in enumerate(tmp_ct):
        status = cv2.imwrite(os.path.join(tmp_ct_save_dir, f"IM{idx:06d}.png"), each_ct_slice)
    tmp_segs = np.sort(glob.glob(os.path.join(os.path.dirname(each_ct), "segmentations", "*.nii.gz")))
    assert len(tmp_segs) == 117
    tmp_seg_single_channel = None
    for seg_idx, each_seg in enumerate(tmp_segs):
        tmp_each_seg = load_and_reorient_nifti(each_seg)
        if not isinstance(tmp_seg_single_channel, np.ndarray):
            tmp_seg_single_channel = np.zeros_like(tmp_each_seg)
        tmp_seg_single_channel[tmp_each_seg == 1] = seg_idx + 1 # add 1 to avoid assigning class to 0
    tmp_seg_save_dir = os.path.join(save_dir, study_id, "segmentations")
    os.makedirs(tmp_seg_save_dir, exist_ok=True)
    for idx, each_seg_slice in enumerate(tmp_seg_single_channel):
        status = cv2.imwrite(os.path.join(tmp_seg_save_dir, f"IM{idx:06d}.png"), each_seg_slice)
