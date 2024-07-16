import albumentations as A
import cv2
import glob
import numpy as np
import os
import pandas as pd
import pydicom

from tqdm import tqdm


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
    instances = np.asarray([int(d.InstanceNumber) for d in dicoms])
    positions = np.asarray([float(d.ImagePositionPatient[plane]) for d in dicoms])
    # if reverse_sort=False, then increasing array index will be from RIGHT->LEFT and CAUDAL->CRANIAL
    # thus we do reverse_sort=True for axial so increasing array index is craniocaudal
    idx = np.argsort(-positions if reverse_sort else positions)
    ipp = np.asarray([d.ImagePositionPatient for d in dicoms]).astype("float")[idx]
    array_shapes = np.vstack([d.pixel_array.shape for d in dicoms])
    h, w = array_shapes[:, 0].max(), array_shapes[:, 1].max()
    # Sometimes the arrays are not all the same shape, pad if necessary
    resizer = A.Resize(height=h, width=w, p=1)
    array = [resizer(image=d.pixel_array.astype("float32"))["image"] for d in dicoms]
    array = np.stack(array)
    array = array[idx]
    return convert_to_8bit(array), instances[idx]


DATA_DIR = "/mnt/stor/datasets/rsna-2024-lumbar-spine-degenerative-classification/"
SAVE_DIR = os.path.join(DATA_DIR, "train_pngs_v2")

all_series = glob.glob(os.path.join(DATA_DIR, "train_images/*/*"))
description_df = pd.read_csv(os.path.join(DATA_DIR, "train_series_descriptions.csv"))
description_dict = {row.series_id: row.series_description for row in description_df.itertuples()}

series_df = pd.DataFrame({"series_folder": all_series})
series_df["study_id"] = series_df.series_folder.apply(lambda x: x.split("/")[-2]).astype("int")
series_df["series_id"] = series_df.series_folder.apply(lambda x: x.split("/")[-1]).astype("int")

failed = []
total_series = len(series_df.series_id.unique())
for each_series, tmp_series_df in tqdm(series_df.groupby("series_id"), total=total_series):
    try:
        study_id = tmp_series_df.study_id.iloc[0]
        tmp_save_dir = os.path.join(SAVE_DIR, str(study_id), str(each_series))
        os.makedirs(tmp_save_dir, exist_ok=True)
        stack, instances = load_dicom_stack([tmp_series_df.series_folder.iloc[0]], plane=description_dict[each_series].split()[0])
        for idx, (each_slice, each_instance) in enumerate(zip(stack, instances)):
            sts = cv2.imwrite(os.path.join(tmp_save_dir, f"IM{idx:06d}_INST{each_instance:06d}.png"), each_slice)
    except Exception as e:
        print(f"FAILED {each_series}: {e}")
        failed.append(each_series)

with open("failed.txt", "w") as f:
    for failure in failed:
        _ = f.write(f"{failure}\n")
