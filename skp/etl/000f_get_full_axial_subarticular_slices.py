import albumentations as A
import cv2
import glob
import numpy as np
import os
import pandas as pd
import pydicom
import sys
sys.path.insert(0, "../../skp")
import torch

from collections import defaultdict
from importlib import import_module
from tqdm import tqdm


def load_model_fold_dict(checkpoint_dict, cfg):
    model_dict = {}
    cfg.pretrained = False
    for fold, checkpoint_path in checkpoint_dict.items():
        print(f"Loading weights from {checkpoint_path} ...")
        wts = torch.load(checkpoint_path)["state_dict"]
        wts = {k.replace("model.", ""): v for k, v in wts.items()}
        model = import_module(f"models.{cfg.model}").Net(cfg)
        model.load_state_dict(wts)
        model = model.eval().cuda()
        model_dict[fold] = model
    return model_dict


def get_image_plane(vals):
    vals = [round(v) for v in vals]
    plane = np.cross(vals[:3], vals[3:6])
    plane = [abs(x) for x in plane]
    return np.argmax(plane) # 0- sagittal, 1- coronal, 2- axial


def convert_to_8bit(x):
    lower, upper = np.percentile(x, (1, 99))
    x = np.clip(x, lower, upper)
    x = x - np.min(x)
    x = x / np.max(x) 
    return (x * 255).astype("uint8")


def load_dicom_stack(dicom_folder_list, plane, reverse_sort=False):
    # Some axial T2 series are broken up into segments
    # Thus we would pass a list of dicom folders and get all the DICOMs from those folders
    # So we can return an array which is sorted by position
    dicom_files = []
    for dicom_folder in dicom_folder_list:
        dicom_files.extend(glob.glob(os.path.join(dicom_folder, "*.dcm")))
    dicoms = [pydicom.dcmread(f) for f in dicom_files]
    # There was one axial T2 study where orientation was coronal but when I checked the images they were axial
    # So we should probably just trust the series description rather than determine orientation ourselves
    # planes = [get_image_plane(d.ImageOrientationPatient) for d in dicoms]
    plane = {"sagittal": 0, "coronal": 1, "axial": 2}[plane.lower()]
    # dicoms = [d for d, p in zip(dicoms, planes) if p == expected_plane]
    instances = [int(d.InstanceNumber) for d in dicoms]
    positions = np.asarray([float(d.ImagePositionPatient[plane]) for d in dicoms])
    # if reverse_sort=False, then increasing array index will be from RIGHT->LEFT and CAUDAL->CRANIAL
    # thus we do reverse_sort=True for axial so increasing array index is craniocaudal
    idx = np.argsort(-positions if reverse_sort else positions)
    ipp = np.asarray([d.ImagePositionPatient for d in dicoms]).astype("float")[idx]
    array_shapes = np.vstack([d.pixel_array.shape for d in dicoms])
    h, w = array_shapes[:, 0].max(), array_shapes[:, 1].max()
    # Sometimes the arrays are not all the same shape, pad if necessary
    padder = A.PadIfNeeded(min_height=h, min_width=w, p=1, border_mode=cv2.BORDER_CONSTANT, value=0)
    array = [padder(image=d.pixel_array.astype("float32"))["image"] for d in dicoms]
    array = np.stack(array)
    array = array[idx]
    return convert_to_8bit(array), ipp, np.asarray(dicoms[0].PixelSpacing).astype("float")


def convert_array_to_submission_df(preds, condition, study_id):
    preds = np.concatenate(preds)
    levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
    grades = ["normal_mild", "moderate", "severe"]
    # assumes preds are in order from L1-L2, L2-L3, ..., L5-S1
    assert preds.shape == (5, 3), f"preds.shape is {preds.shape}"
    row_id_list = [f"{study_id}_{condition}_{l}" for l in levels]
    pred_df = pd.DataFrame(preds)
    pred_df.columns = grades
    pred_df["row_id"] = row_id_list
    return pred_df[["row_id"] + grades]


def get_3_channel_indices(ch2, num_images):
    ch1 = max(0, ch2 - 1)
    ch3 = min(num_images - 1, ch2 + 1) # subtract 1 from num_images since array is 0-indexed
    return [ch1, ch2, ch3]


def crop_square_around_center(img, xc, yc, size_factor=0.15):
    h, w = size_factor * img.shape[0], size_factor * img.shape[1]
    x1, y1 = xc - w / 2, yc - h / 2
    x2, y2 = x1 + w, y1 + h
    x1, y1, x2, y2 = [int(_) for _ in [x1, y1, x2, y2]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    return img[y1:y2, x1:x2]


def save_list_of_images(image_list, study_id, laterality, save_dir):
    levels = ["L1_L2", "L2_L3", "L3_L4", "L4_L5", "L5_S1"]
    filenames = [f"{study_id}_{laterality}_{lvl}.png" if laterality != "" else f"{study_id}_{lvl}.png" for lvl in levels]
    assert len(image_list) == len(levels)
    os.makedirs(save_dir, exist_ok=True)
    for fname, img in zip(filenames, image_list):
        if not isinstance(img, np.ndarray):
            continue
        status = cv2.imwrite(os.path.join(save_dir, fname), img)


###########################
# LOAD CONFIGS AND MODELS #
###########################
cfg_file = "cfg_identify_subarticular_slices_with_level"
cfg = import_module(f"configs.{cfg_file}").cfg
checkpoint_dict = {
    0: "../../skp/experiments/cfg_identify_subarticular_slices_with_level/a0eaf12e/fold0/checkpoints/last.ckpt",
    1: "../../skp/experiments/cfg_identify_subarticular_slices_with_level/51176343/fold1/checkpoints/last.ckpt",
    2: "../../skp/experiments/cfg_identify_subarticular_slices_with_level/4b2e01b9/fold2/checkpoints/last.ckpt",
    3: "../../skp/experiments/cfg_identify_subarticular_slices_with_level/0a8cbac8/fold3/checkpoints/last.ckpt",
    4: "../../skp/experiments/cfg_identify_subarticular_slices_with_level/50ac81cd/fold4/checkpoints/last.ckpt"
}

subarticular_slice_finder_model_2d = {
    "cfg": cfg,
    "models": load_model_fold_dict(checkpoint_dict, cfg)
}

############
# PIPELINE #
############
folds_df = pd.read_csv("../../data/folds_cv5.csv")
study_id_fold_dict = {row.study_id: row.fold for row in folds_df.itertuples()}
description_df = pd.read_csv("../../data/train_series_descriptions.csv")
study_series_id_description_dict = {f"{row.study_id}-{row.series_id}": row.series_description for row in description_df.itertuples()}
dicom_dir = "../../data/train_images/"
save_dir = "../../data/train_subarticular_full_slices/"
os.makedirs(save_dir, exist_ok=True)

levels = ["L1", "L2", "L3", "L4", "L5", "S1"]
levels_dict = {ii: lvl for ii, lvl in enumerate(levels)}

for study_id, fold in tqdm(study_id_fold_dict.items(), total=len(study_id_fold_dict)):
    series = glob.glob(os.path.join(dicom_dir, str(study_id), "*"))
    series_path_dict = defaultdict(list)
    for each_series in series:
        series_path_dict[study_series_id_description_dict[f"{study_id}-{os.path.basename(each_series)}"]].append(each_series)
    AX_T2_AVAILABLE = len(series_path_dict["Axial T2"]) > 0
    if AX_T2_AVAILABLE:
        ax_t2 = series_path_dict["Axial T2"]
        ax_t2, ax_t2_pos, ax_t2_pix = load_dicom_stack(ax_t2, plane="axial", reverse_sort=True)
    # dentify subarticular slices
    if AX_T2_AVAILABLE:
        ax_t2_torch = np.stack([subarticular_slice_finder_model_2d["cfg"].val_transforms(image=img)["image"] for img in ax_t2])
        ax_t2_torch = torch.from_numpy(ax_t2_torch).unsqueeze(1).cuda()
        with torch.inference_mode():
            subout = subarticular_slice_finder_model_2d["models"][fold]({"x": ax_t2_torch})["logits"].sigmoid().cpu().numpy()
        level_preds = subout[:, 1:]
        subart_preds = subout[:, 0]
        assigned_levels = [levels_dict[ii] for ii in np.argmax(level_preds, axis=1)]
        # Get the first occurrence of L2-S1, then subtract 1 because the subarticular slice should be the LAST slice of a given level
        intervertebral_spaces = []
        for lvl in levels[1:]:
            try:
                intervertebral_spaces.append(assigned_levels.index(lvl) - 1)
            except ValueError:
                intervertebral_spaces.append(None)
        target_axial_slices = [
            ax_t2[get_3_channel_indices(ii, num_images=len(ax_t2))].transpose(1, 2, 0) if isinstance(ii, int) else None 
            for ii in intervertebral_spaces
        ]
    assert len(target_axial_slices) == 5
    for each_level, each_slice in zip(["L1_L2", "L2_L3", "L3_L4", "L4_L5", "L5_S1"], target_axial_slices):
        if not isinstance(each_slice, np.ndarray):
            continue
        tmp_savefile = os.path.join(save_dir, f"{study_id}_{each_level}.png")
        status = cv2.imwrite(tmp_savefile, each_slice)

