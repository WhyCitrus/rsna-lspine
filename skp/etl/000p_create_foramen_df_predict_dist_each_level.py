import numpy as np
import pandas as pd


df = pd.read_csv("../../data/train_label_coordinates.csv")
foramen_df = df.loc[df.condition.apply(lambda x: "Foraminal" in x)]

meta_df = pd.read_csv("../../data/dicom_metadata.csv")
meta_df = meta_df.loc[meta_df.series_id.isin(foramen_df.series_id.tolist())]

levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
col_names = [f"rt_{_}" for _ in levels] + [f"lt_{_}" for _ in levels]

new_df_list = []
for series_id, series_df in meta_df.groupby("series_id"):
	series_df = series_df.sort_values("ImagePositionPatient0", ascending=False)
	foramen_series_df = foramen_df.loc[foramen_df.series_id == series_id]
	# skip if not full complement of right and left for each level
	if len(np.unique(foramen_series_df.condition + foramen_series_df.level)) != 10:
		continue
	instance_to_position = {row.instance_number: row.ImagePositionPatient0 for row in series_df.itertuples()}
	foramen_slice_dict = {}
	foramen_series_df = foramen_series_df.sort_values(["level", "condition"])
	for level, level_df in foramen_series_df.groupby("level"):
		foramen_slice_dict["rt_" + level.replace("/", "_").lower()] = instance_to_position[level_df.instance_number.values[1]]
		foramen_slice_dict["lt_" + level.replace("/", "_").lower()] = instance_to_position[level_df.instance_number.values[0]]
	max_dist = series_df.ImagePositionPatient0.max() - series_df.ImagePositionPatient0.min()
	assert max_dist > 0
	for c in col_names:
		# rescale by max distance ... may not be the right decision, but we'll see
		series_df[c] = (series_df.ImagePositionPatient0 - foramen_slice_dict[c]) / max_dist
		# no rescale
		series_df[c + "_no_rescale"] = series_df.ImagePositionPatient0 - foramen_slice_dict[c]
	new_df_list.append(series_df)

new_df = pd.concat(new_df_list)
folds_df = pd.read_csv("../../data/folds_cv5.csv")
new_df = new_df.merge(folds_df, on="study_id")

new_df["pngfile"] = new_df.study_id.astype("str") + "/" + new_df.series_id.astype("str") + "/" + new_df.instance_number.apply(lambda x: f"IM{x:06d}.png")

new_df.to_csv("../../data/train_foramen_distance_each_level.csv", index=False)
