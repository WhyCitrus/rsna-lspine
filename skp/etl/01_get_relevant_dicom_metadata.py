import glob
import numpy as np
import os
import pandas as pd
import pydicom

from collections import defaultdict
from tqdm import tqdm


def get_dicom_attribute(dcm, att):
    try:
        if att in ["PixelSpacing", "ImagePositionPatient", "ImageOrientationPatient"]:
            return [float(_) for _ in getattr(dcm, att)]
        return getattr(dcm, att)
    except:
        if att == "PixelSpacing": 
            return None, None
        elif att == "ImagePositionPatient":
            return None, None, None
        else:
            return None
    
    
def get_image_plane(vals):
    vals = [round(v) for v in vals]
    plane = np.cross(vals[:3], vals[3:6])
    plane = [abs(x) for x in plane]
    return np.argmax(plane) # 0- sagittal, 1- coronal, 2- axial


DATA_DIR = "/mnt/stor/datasets/rsna-2024-lumbar-spine-degenerative-classification/"
all_dicoms = glob.glob(os.path.join(DATA_DIR, "train_images/*/*/*.dcm"))

dicom_dict = defaultdict(list)
image_plane_dict = {0: "SAG", 1: "COR", 2: "AX"}

for each_dicom in tqdm(all_dicoms):
    dcm = pydicom.dcmread(each_dicom)
    dicom_dict["pid"].append(get_dicom_attribute(dcm, "PatientID"))
    dicom_dict["study_id"].append(get_dicom_attribute(dcm, "StudyInstanceUID"))
    dicom_dict["series_id"].append(get_dicom_attribute(dcm, "SeriesInstanceUID").split(".")[1])
    dicom_dict["instance_number"].append(get_dicom_attribute(dcm, "InstanceNumber"))
    dicom_dict["filename"].append(os.path.basename(each_dicom))
    dicom_dict["filepath"].append("/".join(each_dicom.split("/")[4:]))
    dicom_dict["rows"].append(get_dicom_attribute(dcm, "Rows"))
    dicom_dict["cols"].append(get_dicom_attribute(dcm, "Columns"))
    pixel_spacing = get_dicom_attribute(dcm, "PixelSpacing")
    dicom_dict["PixelSpacing0"].append(pixel_spacing[0])
    dicom_dict["PixelSpacing1"].append(pixel_spacing[1])
    image_position = get_dicom_attribute(dcm, "ImagePositionPatient")
    dicom_dict["ImagePositionPatient0"].append(image_position[0])
    dicom_dict["ImagePositionPatient1"].append(image_position[1])
    dicom_dict["ImagePositionPatient2"].append(image_position[2])
    image_orientation = get_dicom_attribute(dcm, "ImageOrientationPatient")
    if not isinstance(image_orientation, type(None)):
        image_orientation = get_image_plane(image_orientation)
        dicom_dict["ImagePlane"].append(image_plane_dict[image_orientation])
        dicom_dict["SliceLocation"].append(image_position[image_orientation])
    else:
        dicom_dict["ImagePlane"].append(image_orientation)
        dicom_dict["SliceLocation"].append(None)

dicom_df = pd.DataFrame(dicom_dict)
dicom_df.head()

dicom_df.to_csv(os.path.join(DATA_DIR, "dicom_metadata.csv"), index=False)
