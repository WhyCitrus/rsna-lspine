import glob
import numpy as np
import os
import pandas as pd 

from PIL import Image


def get_condition(string):
	string = string.lower()
	for condition in ["spinal", "foraminal", "subarticular"]:
		if condition in string:
			return condition


df = pd.read_csv("../../data/train.csv")

df_list = []
for c in df.columns:
	if c == "study_id":
		continue
	tmp_df = df[["study_id", c]].copy()
	tmp_df["laterality"] = c.split("_")[0][0].upper() if "subarticular" in c.lower() or "foraminal" in c.lower() else None
	tmp_df["level"] = "_".join(c.split("_")[-2:]).upper()
	tmp_df.columns = ["study_id", "grade", "laterality", "level"]
	tmp_df["condition"] = get_condition(c)
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

df["sample_weight"] = 1
df.loc[df.moderate == 1, "sample_weight"] = 2
df.loc[df.severe == 1, "sample_weight"] = 4

df.to_csv("../../data/train_narrow.csv", index=False)
