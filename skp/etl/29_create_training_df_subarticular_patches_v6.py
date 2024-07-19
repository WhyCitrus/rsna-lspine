import glob
import numpy as np
import os
import pandas as pd 

from PIL import Image


df = pd.read_csv("../../data/train.csv")
df = df[["study_id"] + [c for c in df.columns if "subarticular" in c]]
df = df.fillna("Missing")

df_list = []
for c in df.columns:
	if c == "study_id":
		continue
	tmp_df = df[["study_id", c]].copy()
	tmp_df["laterality"] = c.split("_")[0][0].upper()
	tmp_df["level"] = "_".join(c.split("_")[-2:]).upper()
	tmp_df.columns = ["study_id", "grade", "laterality", "level"]
	df_list.append(tmp_df)

df = pd.concat(df_list)
labels_map = {
	"Normal/Mild": 0, "Moderate": 1, "Severe": 2
}
df["grade"] = df["grade"].map(labels_map)
df = df.loc[~df.grade.isna()]
df["normal_mild"] = (df.grade == 0).astype("int")
df["moderate"] = (df.grade == 1).astype("int")
df["severe"] = (df.grade == 2).astype("int")

all_images = glob.glob("../../data/train_axial_subarticular_crops_3ch_v6/*/*.png")
image_df = pd.DataFrame({"filepath": all_images})
image_df["filepath"] = image_df.filepath.apply(lambda x: x.replace("../../data/train_axial_subarticular_crops_3ch_v6/", ""))
image_df["study_id"] = image_df.filepath.apply(lambda x: x.split("/")[-2]).astype("int")
image_df["laterality"] = image_df.filepath.apply(lambda x: os.path.basename(x)[:1])
image_df["level"] = image_df.filepath.apply(lambda x: os.path.basename(x).replace(".png", "")[-5:])

df = df.merge(image_df, on=["study_id", "laterality", "level"])
folds_df = pd.read_csv("../../data/folds_cv5.csv")
df = df.merge(folds_df, on="study_id")

df.to_csv("../../data/train_subarticular_crops_v6.csv", index=False)
