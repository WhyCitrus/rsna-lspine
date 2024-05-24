import glob
import os
import pandas as pd


df = pd.read_csv("meta.csv", sep=";")
test_study = glob.glob("*")[-1]
segmentations = glob.glob(os.path.join(test_study, "segmentations", "*.nii.gz"))

segmentation_labels = [os.path.basename(_).replace(".nii.gz", "") for _ in segmentations]