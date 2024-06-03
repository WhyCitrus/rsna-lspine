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
cfg_file = "cfg_predict_sagittal_foramina_coords"
cfg = import_module(f"configs.{cfg_file}").cfg
checkpoint_dict = {
    0: "../../skp/experiments/cfg_predict_sagittal_foramina_coords/312b2196/fold0/checkpoints/last.ckpt",
    1: "../../skp/experiments/cfg_predict_sagittal_foramina_coords/19164216/fold1/checkpoints/last.ckpt",
    2: "../../skp/experiments/cfg_predict_sagittal_foramina_coords/ed7c9c8a/fold2/checkpoints/last.ckpt",
    3: "../../skp/experiments/cfg_predict_sagittal_foramina_coords/b0a5b7d6/fold3/checkpoints/last.ckpt",
    4: "../../skp/experiments/cfg_predict_sagittal_foramina_coords/c4217cb1/fold4/checkpoints/last.ckpt"
}

foramina_localization_model_3d = {
    "cfg": cfg,
    "models": load_model_fold_dict(checkpoint_dict, cfg)
}

##
cfg_file = "cfg_predict_sagittal_canal_coords"
cfg = import_module(f"configs.{cfg_file}").cfg
checkpoint_dict = {
    0: "../../skp/experiments/cfg_predict_sagittal_canal_coords/016642e2/fold0/checkpoints/last.ckpt",
    1: "../../skp/experiments/cfg_predict_sagittal_canal_coords/97cd59e2/fold1/checkpoints/last.ckpt",
    2: "../../skp/experiments/cfg_predict_sagittal_canal_coords/a3b02859/fold2/checkpoints/last.ckpt",
    3: "../../skp/experiments/cfg_predict_sagittal_canal_coords/c9e60103/fold3/checkpoints/last.ckpt",
    4: "../../skp/experiments/cfg_predict_sagittal_canal_coords/74592ed9/fold4/checkpoints/last.ckpt"
}

canal_localization_model_3d = {
    "cfg": cfg,
    "models": load_model_fold_dict(checkpoint_dict, cfg)
}

##
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

##
cfg_file = "cfg_axial_subarticular_coords"
cfg = import_module(f"configs.{cfg_file}").cfg
checkpoint_dict = {
    0: "../../skp/experiments/cfg_axial_subarticular_coords/0427e248/fold0/checkpoints/last.ckpt",
    1: "../../skp/experiments/cfg_axial_subarticular_coords/d0ab9f20/fold1/checkpoints/last.ckpt",
    2: "../../skp/experiments/cfg_axial_subarticular_coords/93cbb14f/fold2/checkpoints/last.ckpt",
    3: "../../skp/experiments/cfg_axial_subarticular_coords/2a89cdd3/fold3/checkpoints/last.ckpt",
    4: "../../skp/experiments/cfg_axial_subarticular_coords/fd2f35d2/fold4/checkpoints/last.ckpt"
}

subarticular_localization_model_2d = {
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
save_dir = "../../data/train_generated_crops/"

levels = ["L1", "L2", "L3", "L4", "L5", "S1"]
levels_dict = {ii: lvl for ii, lvl in enumerate(levels)}

for study_id, fold in tqdm(study_id_fold_dict.items(), total=len(study_id_fold_dict)):
    series = glob.glob(os.path.join(dicom_dir, str(study_id), "*"))
    series_path_dict = defaultdict(list)
    for each_series in series:
        series_path_dict[study_series_id_description_dict[f"{study_id}-{os.path.basename(each_series)}"]].append(each_series)
    SAG_T1_AVAILABLE = len(series_path_dict["Sagittal T1"]) > 0
    SAG_T2_AVAILABLE = len(series_path_dict["Sagittal T2/STIR"]) > 0
    AX_T2_AVAILABLE = len(series_path_dict["Axial T2"]) > 0
    # A few studies had multiple sagittal T1 series
    # Upon manual review, it seems that they were all duplicates of each other
    # So we should be fine just taking any of them
    if SAG_T1_AVAILABLE:
        sag_t1 = series_path_dict["Sagittal T1"][0]
        sag_t1, sag_t1_pos, sag_t1_pix = load_dicom_stack([sag_t1], expected_plane="sagittal")
    # There were no studies with multiple sagittal T2 series
    # Though if there are in the test set, I assume the above would also apply
    if SAG_T2_AVAILABLE:
        sag_t2 = series_path_dict["Sagittal T2/STIR"][0]
        sag_t2, sag_t2_pos, sag_t2_pix = load_dicom_stack([sag_t2], expected_plane="sagittal")
    # Some studies split axial T2s into segments 
    # So we would need to load all the available axial series
    if AX_T2_AVAILABLE:
        ax_t2 = series_path_dict["Axial T2"]
        ax_t2, ax_t2_pos, ax_t2_pix = load_dicom_stack(ax_t2, expected_plane="axial", reverse_sort=True)
    # 1- Identify foramina coords (sagittal T1)
    if SAG_T1_AVAILABLE:
        sag_t1_torch = foramina_localization_model_3d["cfg"].val_transforms({"image": np.expand_dims(sag_t1, axis=0)})["image"]
        with torch.inference_mode():
            out = foramina_localization_model_3d["models"][fold]({"x": sag_t1_torch.unsqueeze(0).cuda()})["logits"].sigmoid().cpu().numpy()[0]
        out[:10] = out[:10] * sag_t1.shape[2]
        out[10:20] = out[10:20] * sag_t1.shape[1]
        out[20:] = out[20:] * sag_t1.shape[0]
        out = out.astype("int")
        lt, rt = np.stack([out[:5], out[10:15], out[20:25]], axis=0), np.stack([out[5:10], out[15:20], out[25:]], axis=0)
        lt_foramen_crops, rt_foramen_crops = [], []
        for level in range(5):
            # LEFT
            ch1, ch2, ch3 = get_3_channel_indices(ch2=lt[2, level], num_images=sag_t1.shape[0])
            tmp_slice = sag_t1[[ch1, ch2, ch3]].transpose(1, 2, 0)
            cropped_foramen = crop_square_around_center(img=tmp_slice, xc=lt[0, level], yc=lt[1, level], size_factor=0.15)
            lt_foramen_crops.append(cropped_foramen)
            # RIGHT
            ch1, ch2, ch3 = get_3_channel_indices(ch2=rt[2, level], num_images=sag_t1.shape[0])
            tmp_slice = sag_t1[[ch1, ch2, ch3]].transpose(1, 2, 0)
            cropped_foramen = crop_square_around_center(img=tmp_slice, xc=rt[0, level], yc=rt[1, level], size_factor=0.15)
            rt_foramen_crops.append(cropped_foramen)
        save_list_of_images(lt_foramen_crops, study_id, laterality="L", save_dir=os.path.join(save_dir, "foraminal"))
        save_list_of_images(rt_foramen_crops, study_id, laterality="R", save_dir=os.path.join(save_dir, "foraminal"))
    # 2- Identify spinal canal coords (sagittal T2)
    if SAG_T2_AVAILABLE:
        sag_t2_torch = canal_localization_model_3d["cfg"].val_transforms({"image": np.expand_dims(sag_t2, axis=0)})["image"]
        with torch.inference_mode():
            canal_out = canal_localization_model_3d["models"][fold]({"x": sag_t2_torch.unsqueeze(0).cuda()})["logits"].sigmoid().cpu().numpy()[0]
        canal_out[:5] = canal_out[:5] * sag_t2.shape[2]
        canal_out[5:10] = canal_out[5:10] * sag_t2.shape[1]
        canal_out[10:] = canal_out[10:] * sag_t2.shape[0]
        canal_out = canal_out.astype("int")
        canal_out = np.stack([canal_out[:5], canal_out[5:10], canal_out[10:]], axis=0)
        canal_crops = []
        for level in range(5):
            ch1, ch2, ch3 = get_3_channel_indices(ch2=canal_out[2, level], num_images=sag_t2.shape[0])
            tmp_slice = sag_t2[[ch1, ch2, ch3]].transpose(1, 2, 0)
            cropped_canal = crop_square_around_center(img=tmp_slice, xc=canal_out[0, level], yc=canal_out[1, level], size_factor=0.15)
            canal_crops.append(cropped_canal)
        save_list_of_images(canal_crops, study_id, laterality="", save_dir=os.path.join(save_dir, "spinal"))
    # 3- Identify subarticular slices
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
    # 4- Identify subarticular coords (axial T2)
        lt_sub_crops, rt_sub_crops = [], []
        for each_slice in target_axial_slices:
            if isinstance(each_slice, type(None)):
                lt_sub_crops.append(None)
                rt_sub_crops.append(None)
                continue
            slice_torch = torch.from_numpy(subarticular_localization_model_2d["cfg"].val_transforms(image=each_slice)["image"].transpose(2, 0, 1))
            with torch.inference_mode():
                out = subarticular_localization_model_2d["models"][fold]({"x": slice_torch.unsqueeze(0).cuda()})["logits"].sigmoid().cpu().numpy()[0]
            out[[0, 2]] *= each_slice.shape[1]
            out[[1, 3]] *= each_slice.shape[0]
            out = out.astype("int")
            lt_x, lt_y, rt_x, rt_y = out
            # LEFT
            cropped_subarticular = crop_square_around_center(img=each_slice, xc=lt_x, yc=lt_y, size_factor=0.15)
            lt_sub_crops.append(cropped_subarticular)
            # RIGHT 
            cropped_subarticular = crop_square_around_center(img=each_slice, xc=rt_x, yc=rt_y, size_factor=0.15)
            rt_sub_crops.append(cropped_subarticular)
        save_list_of_images(lt_sub_crops, study_id, laterality="L", save_dir=os.path.join(save_dir, "subarticular"))
        save_list_of_images(rt_sub_crops, study_id, laterality="R", save_dir=os.path.join(save_dir, "subarticular"))
