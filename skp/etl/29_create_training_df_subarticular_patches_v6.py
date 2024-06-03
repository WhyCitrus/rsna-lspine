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

coords_df = pd.read_csv("../../data/train_label_coordinates.csv")
coords_df = coords_df.loc[coords_df.condition.apply(lambda x: "Subarticular" in x)]
coords_df["study_side_level"] = coords_df.study_id.astype("str") + "_" + coords_df.condition.apply(lambda x: x[:1]) + "_" + coords_df.level.apply(lambda x: x.replace("/", "_"))
df["study_side_level"] = df.study_id.astype("str") + "_" + df.laterality + "_" + df["level"]

df = df.loc[df.study_side_level.isin(coords_df.study_side_level.tolist())]
df["filepath"] = df.study_id.astype("str") + "/" + df.laterality + "T_" + df.level + ".png"
folds_df = pd.read_csv("../../data/folds_cv5.csv")
df = df.merge(folds_df, on="study_id")

df.to_csv("../../data/train_subarticular_crops_v4.csv", index=False)
