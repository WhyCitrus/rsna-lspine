import numpy as np
import pandas as pd

from tqdm import tqdm


df = pd.read_csv("../../data/train_label_coordinates.csv")
foramen_df = df.loc[df.condition.apply(lambda x: "Foraminal" in x)]

meta_df = pd.read_csv("../../data/dicom_metadata.csv")
meta_df = meta_df.loc[meta_df.series_id.isin(foramen_df.series_id.tolist())]

levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
col_names = [f"rt_{_}" for _ in levels] + [f"lt_{_}" for _ in levels]

# The following studies are incorrectly labeled (e.g., left and right sides are swapped or one side is labeled left and right)
incorrectly_labeled = [
	3742728457, 364930790, 2410494888, 2135829458, 1879696087, 3156269631, 1085426528, 3781188430, 3495818564,
	1395773918, 757619082, 2626030939, 2530679352, 2388577668, 1647904243, 796739553, 2662989538
]

new_df_list = []
for series_id, series_df in tqdm(meta_df.groupby("series_id"), total=len(meta_df.series_id.unique())):
	series_df = series_df.sort_values("ImagePositionPatient0", ascending=False)
	foramen_series_df = foramen_df.loc[foramen_df.series_id == series_id]
	# skip if not full complement of right and left for each level
	if len(np.unique(foramen_series_df.condition + foramen_series_df.level)) != 10:
		continue
	if series_df.study_id.values[0] in incorrectly_labeled:
		continue
	instance_to_position = {row.instance_number: row.ImagePositionPatient0 for row in series_df.itertuples()}
	foramen_slice_dict = {}
	foramen_coord_dict = {}
	foramen_series_df = foramen_series_df.sort_values(["level", "condition"])
	for level, level_df in foramen_series_df.groupby("level"):
		foramen_slice_dict["rt_" + level.replace("/", "_").lower()] = instance_to_position[level_df.instance_number.values[1]]
		foramen_slice_dict["lt_" + level.replace("/", "_").lower()] = instance_to_position[level_df.instance_number.values[0]]
		foramen_coord_dict["rt_" + level.replace("/", "_").lower()] = (level_df.x.values[1], level_df.y.values[1])
		foramen_coord_dict["lt_" + level.replace("/", "_").lower()] = (level_df.x.values[0], level_df.y.values[0])
	max_dist = series_df.ImagePositionPatient0.max() - series_df.ImagePositionPatient0.min()
	assert max_dist > 0
	for c in col_names:
		# rescale by max distance ...
		series_df[c] = (series_df.ImagePositionPatient0 - foramen_slice_dict[c]) / max_dist
		# no rescale
		series_df[c + "_no_rescale"] = series_df.ImagePositionPatient0 - foramen_slice_dict[c]
		# add in rescaled coordinates of the foramen
		series_df[c + "_foramen_coord_x"] = foramen_coord_dict[c][0]
		series_df[c + "_foramen_coord_y"] = foramen_coord_dict[c][1]
	new_df_list.append(series_df)

new_df = pd.concat(new_df_list)
for c in col_names:
	new_df[c + "_foramen_coord_x"] = new_df[c + "_foramen_coord_x"] / new_df.cols
	new_df[c + "_foramen_coord_y"] = new_df[c + "_foramen_coord_y"] / new_df.rows

folds_df = pd.read_csv("../../data/folds_cv5.csv")
new_df = new_df.merge(folds_df, on="study_id")

new_df["pngfile"] = new_df.study_id.astype("str") + "/" + new_df.series_id.astype("str") + "/" + new_df.instance_number.apply(lambda x: f"IM{x:06d}.png")

new_df.to_csv("../../data/train_foramen_distance_each_level_with_foramen_coords.csv", index=False)
