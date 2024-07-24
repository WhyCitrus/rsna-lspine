import numpy as np
import pandas as pd

from tqdm import tqdm


df = pd.read_csv("../../data/train_label_coordinates.csv")
subart_df = df.loc[df.condition.apply(lambda x: "Subarticular" in x)]

meta_df = pd.read_csv("../../data/dicom_metadata.csv")
meta_df = meta_df.loc[meta_df.series_id.isin(subart_df.series_id.tolist())]
instance_to_position_index_dict = {}
for series_id, series_df in meta_df.groupby("series_id"):
	series_df = series_df.sort_values("ImagePositionPatient2", ascending=True)
	series_df["position_index"] = np.arange(len(series_df))
	instance_to_position_index_dict.update({f"{series_id}_{row.instance_number}": row.position_index for row in series_df.itertuples()})

levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
col_names = [f"rt_{_}" for _ in levels] + [f"lt_{_}" for _ in levels]

new_df_list = []
for series_id, series_df in tqdm(meta_df.groupby("series_id"), total=len(meta_df.series_id.unique())):
	series_df = series_df.sort_values("ImagePositionPatient2", ascending=False)
	subart_series_df = subart_df.loc[subart_df.series_id == series_id]
	# skip if not full complement of right and left for each level
	if len(np.unique(subart_series_df.condition + subart_series_df.level)) != 10:
		continue
	instance_to_position = {row.instance_number: row.ImagePositionPatient2 for row in series_df.itertuples()}
	subart_slice_dict = {}
	subart_coord_dict = {}
	subart_series_df = subart_series_df.sort_values(["level", "condition"])
	for level, level_df in subart_series_df.groupby("level"):
		subart_slice_dict["rt_" + level.replace("/", "_").lower()] = instance_to_position[level_df.instance_number.values[1]]
		subart_slice_dict["lt_" + level.replace("/", "_").lower()] = instance_to_position[level_df.instance_number.values[0]]
		subart_coord_dict["rt_" + level.replace("/", "_").lower()] = (level_df.x.values[1], level_df.y.values[1])
		subart_coord_dict["lt_" + level.replace("/", "_").lower()] = (level_df.x.values[0], level_df.y.values[0])
	max_dist = series_df.ImagePositionPatient2.max() - series_df.ImagePositionPatient2.min()
	assert max_dist > 0
	for c in col_names:
		# rescale by max distance ...
		series_df[c] = (series_df.ImagePositionPatient2 - subart_slice_dict[c]) / max_dist
		# no rescale
		series_df[c + "_no_rescale"] = series_df.ImagePositionPatient2 - subart_slice_dict[c]
		# add in rescaled coordinates of the subart
		series_df[c + "_subart_coord_x"] = subart_coord_dict[c][0] / series_df.cols
		series_df[c + "_subart_coord_y"] = subart_coord_dict[c][1] / series_df.rows
	# Change coords to -1 if they are not on the right slice or adjacent slices (+/- 1)
	series_df = series_df.reset_index(drop=True)
	for c in col_names:
		correct_slice = np.where(series_df[c] == 0)[0][0]
		correct_slice = np.arange(correct_slice - 1, correct_slice + 2)
		incorrect_slices = list(set(range(len(series_df))) - set(correct_slice))
		series_df.loc[incorrect_slices, f"{c}_subart_coord_x"] = -1
		series_df.loc[incorrect_slices, f"{c}_subart_coord_y"] = -1
	new_df_list.append(series_df)

new_df = pd.concat(new_df_list)

folds_df = pd.read_csv("../../data/folds_cv5.csv")
new_df = new_df.merge(folds_df, on="study_id")

pngfile = []
for row in new_df.itertuples():
	pngfile.append(f"{row.study_id}/{row.series_id}/IM{instance_to_position_index_dict[str(row.series_id) + '_' + str(row.instance_number)]:06d}_INST{row.instance_number:06d}.png")

new_df["pngfile"] = pngfile

new_df.to_csv("../../data/train_subart_distance_each_level_with_subart_coords_and_ignore.csv", index=False)
