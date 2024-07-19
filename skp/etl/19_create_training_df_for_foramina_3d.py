import glob
import numpy as np
import os
import pandas as pd 

from PIL import Image


df = pd.read_csv("../../data/train.csv")
df = df[["study_id"] + [c for c in df.columns if "neural_foraminal" in c]]

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
df = df.loc[~df.grade.isna()]
df["grade"] = df["grade"].map(labels_map).astype("int")
df["normal_mild"] = (df.grade == 0).astype("int")
df["moderate"] = (df.grade == 1).astype("int")
df["severe"] = (df.grade == 2).astype("int")

rt_df = df.loc[df.laterality == "R"]
lt_df = df.loc[df.laterality == "L"]

df = rt_df.merge(lt_df, on=["study_id", "level"], suffixes=("_rt", "_lt"))

foramina_crops = glob.glob("../../data/train_foramina_crops_3d/*/*")
crop_df = pd.DataFrame({"series_folder": foramina_crops})
crop_df["series_folder"] = crop_df.series_folder.apply(lambda x: x.replace("../../data/train_foramina_crops_3d/", ""))
crop_df["study_id"] = crop_df.series_folder.apply(lambda x: x.split("/")[0]).astype("int")
crop_df["series_id"] = crop_df.series_folder.apply(lambda x: x.split("/")[1]).astype("int")

df = df.merge(crop_df, on="study_id")
df["series_folder"] = df.series_folder + "/" + df.level
folds_df = pd.read_csv("../../data/folds_cv5.csv")
df = df.merge(folds_df, on="study_id")

df.to_csv("../../data/train_foramina_crops_3d.csv", index=False)
