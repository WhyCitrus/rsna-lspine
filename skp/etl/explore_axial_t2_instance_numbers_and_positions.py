"""
It appears that sorting by ImagePositionPatient[2] does not always necessarily work.

Presumably instance number would be better? 
"""
import numpy as np
import pandas as pd


def check_equivalence(instances, positions):
	instances = instances - 1
	if (instances == positions).sum() == len(instances):
		return True
	if (instances[::-1] == positions).sum() == len(instances):
		return True
	return False


meta_df = pd.read_csv("../../data/dicom_metadata.csv")
desc_df = pd.read_csv("../../data/train_series_descriptions.csv")
ax_t2_series = desc_df.loc[desc_df.series_description == "Axial T2"]
meta_df = meta_df.loc[meta_df.series_id.isin(ax_t2_series.series_id.tolist())]

not_equal = []
for series_id, series_df in meta_df.groupby("series_id"):
	series_df = series_df.sort_values("ImagePositionPatient2")
	series_df["position_index"] = np.arange(len(series_df))
	series_df["reversed_position_index"] = np.arange(len(series_df))[::-1]
	series_df = series_df.sort_values("instance_number")
	series_df["instance_index"] = series_df.instance_number - 1
	equal = check_equivalence(series_df.instance_number.values, series_df.position_index.values)
	if not equal:
		not_equal.append(series_df)

print(len(not_equal))

not_equal_df = pd.concat(not_equal)
not_equal_df[["series_id", "instance_number", "instance_index", "position_index", "reversed_position_index", "ImagePositionPatient2"]].to_csv("../../data/not_equal_axial_t2_series.csv", index=False)