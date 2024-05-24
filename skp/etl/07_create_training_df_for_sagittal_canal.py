import glob
import numpy as np
import os
import pandas as pd 

from PIL import Image


df = pd.read_csv("../../data/train.csv")
df = df[["study_id"] + [c for c in df.columns if "canal" in c]]

df_list = []
for c in df.columns:
	if c == "study_id":
		continue
	tmp_df = df[["study_id", c]].copy()
	tmp_df["level"] = "_".join(c.split("_")[-2:]).upper()
	tmp_df.columns = ["study_id", "grade", "level"]
	df_list.append(tmp_df)

df = pd.concat(df_list)
labels_map = {
	"Normal/Mild": 0, "Moderate": 1, "Severe": 2
}
df = df.loc[~df.grade.isna()]
df["grade"] = df["grade"].map(labels_map).astype("int")
df["normal_mild"] = (df.grade == 0).astype("int")
df["moderate"] = (df.grade == 1).astype("int")
df["severe"] = (df.grade == 2).astype("int")

print(df.shape)
df.grade.value_counts()

canal_crops = glob.glob("../../data/train_sagittal_canal_crops_3ch/*/*/*.png")
# check image sizes
image_sizes = np.vstack([Image.open(_).size for _ in canal_crops])
print(image_sizes[:, 0].mean(), image_sizes[:, 1].mean())

canal_crops = [_.replace("../../data/train_sagittal_canal_crops_3ch/", "") for _ in canal_crops]

patch_df = pd.DataFrame({"filepath": canal_crops})
patch_df["study_id"] = patch_df.filepath.apply(lambda x: x.split("/")[-3]).astype("int")
patch_df["series_id"] = patch_df.filepath.apply(lambda x: x.split("/")[-2]).astype("int")
patch_df["level"] = patch_df.filepath.apply(lambda x: x.split("/")[-1].replace(".png", ""))

df = df.merge(patch_df, on=["study_id", "level"])
folds_df = pd.read_csv("../../data/folds_cv5.csv")
df = df.merge(folds_df, on="study_id")

df.to_csv("../../data/train_sagittal_canal_crops.csv", index=False)
