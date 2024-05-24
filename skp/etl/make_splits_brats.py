import glob
import os
import pandas as pd

from utils import create_double_cv


studies = glob.glob("/mnt/stor/datasets/brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/*")

df = pd.DataFrame(dict(study_folder=studies))
df["pid"] = df.study_folder.apply(lambda x: os.path.basename(x).split("-")[-2])

cv_df = create_double_cv(df, "pid", 5, 5)
cv_df.to_csv("/mnt/stor/datasets/brats/train_kfold.csv", index=False)
