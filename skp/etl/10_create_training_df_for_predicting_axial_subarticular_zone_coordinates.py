import numpy as np
import pandas as pd

from utils import create_double_cv


df = pd.read_csv("../../data/train_label_coordinates.csv")
df = df.loc[df.condition.apply(lambda x: "Subarticular" in x)]

# What is the largest distance between right and left subarticular stenosis for a given level? 
unilateral = []
distance_dict = {}
for series_id, series_df in df.groupby("series_id"):
	for each_level, level_df in series_df.groupby("level"):
		if len(level_df) != 2:
			unilateral.append(f"{series_id}_{each_level}")
			continue
		distance_dict[f"{series_id}_{each_level}"] = np.abs(level_df.instance_number.values[0] - level_df.instance_number.values[1])

np.unique(list(distance_dict.values()), return_counts=True)
# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10]), array([5459, 3005,  200,   35,   14,   23,   17,    6,    5,    5,    1]))
# vast majority are 2 or under, the one that is 10 seems to be typo

df_list = []
for series_id, series_df in df.groupby("series_id"):
	for each_level, level_df in series_df.groupby("level"):
		if len(level_df) != 2:
			continue
		inst1 = level_df.instance_number.values[0]
		inst2 = level_df.instance_number.values[1] 
		if np.abs(inst1 - inst2) > 2:
			continue
		target_inst = int(np.mean([inst1, inst2]))
		level_df = level_df.sort_values("condition")
		left_x, left_y = level_df["x"].iloc[0], level_df["y"].iloc[0]
		right_x, right_y = level_df["x"].iloc[1], level_df["y"].iloc[1]
		tmp_df = pd.DataFrame({"study_id": level_df.study_id.iloc[0], "series_id": level_df.series_id.iloc[0], "instance_number": target_inst, "left_x": left_x, "left_y": left_y, "right_x": right_x, "right_y": right_y, "level": level_df["level"].iloc[0]}, index=[0])
		df_list.append(tmp_df)

coords_df = pd.concat(df_list)
print(coords_df.shape)

coords_df["filepath"] = coords_df.study_id.astype("str") + "/" + coords_df.series_id.astype("str") + "/" + coords_df.instance_number.apply(lambda x: f"IM{x:06d}.png")
meta_df = pd.read_csv("../../data/dicom_metadata.csv")
coords_df = coords_df.merge(meta_df[["study_id", "series_id", "instance_number", "rows", "cols"]])
coords_df["lt_subarticular_x"] = coords_df.left_x / coords_df["cols"]
coords_df["lt_subarticular_y"] = coords_df.left_y / coords_df["rows"]
coords_df["rt_subarticular_x"] = coords_df.right_x / coords_df["cols"]
coords_df["rt_subarticular_y"] = coords_df.right_y / coords_df["rows"]

coords_df = create_double_cv(coords_df, "study_id", 5, 5)
coords_df.to_csv("../../data/train_predicting_axial_subarticular_coords.csv", index=False)

