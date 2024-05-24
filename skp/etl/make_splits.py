import glob
import pandas as pd

from utils import create_double_cv


studies = glob.glob("/mnt/IDH_Project/Segmented_Images_Reviewed/*/*")

df = pd.DataFrame(dict(study_folder=studies))
df["pid"] = df.study_folder.apply(lambda x: x.split("/")[-2])
df["study_id"] = df.study_folder.apply(lambda x: x.split("/")[-1])
df

cv_df = create_double_cv(df, "pid", 5, 5)
cv_df.to_csv("/home/neurolab/ianpan/train_segmented_images_reviewed_kfold.csv", index=False)
