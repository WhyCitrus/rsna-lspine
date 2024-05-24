import glob
import os
import pandas as pd

from utils import create_double_cv


studies = glob.glob("/mnt/stor/datasets/totalsegmentator/pngs-v201/*")

df = pd.DataFrame(dict(study_folder=studies))
df["pid"] = df.study_folder.apply(lambda x: os.path.basename(x))
df["image_folder"] = df.study_folder.apply(lambda x: os.path.join(x, "images"))
df["segmentation_folder"] = df.study_folder.apply(lambda x: os.path.join(x, "segmentations"))

cv_df = create_double_cv(df, "pid", 5, 5)
cv_df.to_csv("/mnt/stor/datasets/totalsegmentator/train_pngs_kfold.csv", index=False)
