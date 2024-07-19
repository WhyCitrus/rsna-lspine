import glob
import numpy as np
import pandas as pd

from functools import partial


def return_level(position, level_dict):
	for level, level_range in level_dict.items():
		lower, upper = level_range
		if not isinstance(lower, type(None)) and not isinstance(upper, type(None)):
			if position >= lower and position < upper:
				return level
		elif not isinstance(lower, type(None)):
			if position == lower: 
				return level
	return None


df = pd.read_csv("../../data/train_label_coordinates.csv")
df = df.loc[df.condition.apply(lambda x: "Subarticular" in x)]
df["filepath"] = df.study_id.astype(str) + "/" + df.series_id.astype(str) + "/" + df.instance_number.apply(lambda x: f"IM{x:06d}.png")

meta_df = pd.read_csv("../../data/dicom_metadata.csv")
meta_df = meta_df.loc[meta_df.ImagePlane == "AX"] # for some reason there are coronal images for a study?

levels = df.level.unique().tolist()

df_list = []
for series_id, series_df in df.groupby("series_id"):
	level_inferior_margins = {}
	tmp_meta_df = meta_df.loc[meta_df.series_id == series_id]
	tmp_meta_df = tmp_meta_df.sort_values("SliceLocation", ascending=False)
	series_df = series_df.merge(tmp_meta_df[["instance_number", "SliceLocation"]])
	series_df = series_df.sort_values("level", ascending=True)
	for lvl, level_df in series_df.groupby("level"):
		level_inferior_margins[lvl] = level_df.SliceLocation.min()
	level_inferior_margins["T12"] = np.inf
	level_inferior_margins["S2"] = -np.inf
	level_ranges = {}
	level_ranges["L1"] = (level_inferior_margins.get("L1/L2", None), level_inferior_margins.get("T12", None))
	level_ranges["L2"] = (level_inferior_margins.get("L2/L3", None), level_inferior_margins.get("L1/L2", None))
	level_ranges["L3"] = (level_inferior_margins.get("L3/L4", None), level_inferior_margins.get("L2/L3", None))
	level_ranges["L4"] = (level_inferior_margins.get("L4/L5", None), level_inferior_margins.get("L3/L4", None))
	level_ranges["L5"] = (level_inferior_margins.get("L5/S1", None), level_inferior_margins.get("L4/L5", None))
	level_ranges["S1"] = (level_inferior_margins.get("S2", None), level_inferior_margins.get("L5/S1", None))
	assign_level_func = partial(return_level, level_dict=level_ranges)
	tmp_meta_df["assigned_level"] = tmp_meta_df.SliceLocation.map(assign_level_func)
	tmp_meta_df["subarticular_slice_present"] = 0
	tmp_meta_df.loc[tmp_meta_df.instance_number.isin(series_df.instance_number.tolist()), "subarticular_slice_present"] = 1
	df_list.append(tmp_meta_df[["study_id", "series_id", "instance_number", "SliceLocation", "assigned_level", "subarticular_slice_present"]])

new_df = pd.concat(df_list)
new_df = new_df.loc[~new_df.assigned_level.isna()]
for each_level in new_df.assigned_level.unique():
	new_df[each_level] = (new_df.assigned_level == each_level).astype("int")

new_df["filepath"] = new_df.study_id.astype("str") + "/" + new_df.series_id.astype("str") + "/" + new_df.instance_number.apply(lambda x: f"IM{x:06d}.png")
folds_df = pd.read_csv("../../data/folds_cv5.csv")
new_df = new_df.merge(folds_df, on="study_id")

new_df.to_csv("../../data/train_identify_subarticular_slices_with_level.csv", index=False)
