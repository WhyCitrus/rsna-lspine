import numpy as np
import pandas as pd

from tqdm import tqdm


df = pd.read_csv("../../data/train_label_coordinates.csv")
subart_df = df.loc[df.condition.apply(lambda x: "Subarticular" in x)]

subart_df["condition_level"] = subart_df.condition.apply(lambda x: x.split()[0].lower()) + "_" + subart_df.level.apply(lambda x: x.replace("/", "_").lower())
condition_levels = list(subart_df.condition_level.unique())

swap_sides_dict = {k: k.replace("left", "right") if "left" in k else k.replace("right", "left") for k in condition_levels}

new_df_list = []
excluded = []
for series_id, series_df in tqdm(subart_df.groupby("series_id"), total=len(subart_df.series_id.unique())):
	# If all levels are present but some sides are missing
	# Impute the missing side with the contralateral side
	if len(series_df.level.unique()) == 5 and len(series_df.condition_level.unique()) != 10:
		missing_condition_levels = list(set(condition_levels) - set(series_df.condition_level))
		impute_df = series_df.copy()
		impute_df["condition"] = impute_df.condition.apply(lambda x: x.replace("Left", "Right") if "Left" in x else x.replace("Right", "Left"))
		impute_df["condition_level"] = impute_df.condition_level.map(swap_sides_dict)
		impute_df = impute_df.loc[impute_df.condition_level.isin(missing_condition_levels)]
		impute_df["x"] = np.nan
		impute_df["y"] = np.nan
		series_df = pd.concat([series_df, impute_df])
	# If some levels are present but some sides are missing
	# Impute the missing side with the contralateral side
	num_condition_levels = len(series_df.condition_level.unique())
	if len(series_df.level.unique()) < 5 and num_condition_levels < len(series_df.level.unique()) * 2:
		missing_condition_levels = list(set(condition_levels) - set(series_df.condition_level))
		impute_df = series_df.copy()
		impute_df["condition"] = impute_df.condition.apply(lambda x: x.replace("Left", "Right") if "Left" in x else x.replace("Right", "Left"))
		impute_df["condition_level"] = impute_df.condition_level.map(swap_sides_dict)
		impute_df = impute_df.loc[impute_df.condition_level.isin(missing_condition_levels)]
		impute_df["x"] = np.nan
		impute_df["y"] = np.nan
		series_df = pd.concat([series_df, impute_df])
	# If missing levels
	# Just add NaNs for the missing levels
	if len(series_df.level.unique()) < 5:
		missing_condition_levels = list(set(condition_levels) - set(series_df.condition_level))
		impute_df = pd.concat([series_df.iloc[:1].copy()] * len(missing_condition_levels))
		impute_df["condition_level"] = missing_condition_levels
		impute_df["instance_number"] = np.nan
		impute_df["x"] = np.nan
		impute_df["y"] = np.nan
		impute_df["condition"] = impute_df.condition_level.apply(lambda x: "Left Subarticular Stenosis" if "left" in x else "Right Subarticular Stenosis")
		impute_df["level"] = impute_df.condition_level.apply(lambda x: "/".join(x.split("_")[1:]).upper())
		series_df = pd.concat([series_df, impute_df])
	if len(series_df.condition_level.unique()) != 10:
		excluded.append(series_id)
		continue
	new_df_list.append(series_df)

subart_df = pd.concat(new_df_list)

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
	instance_to_position = {row.instance_number: row.ImagePositionPatient2 for row in series_df.itertuples()}
	subart_slice_dict = {}
	subart_coord_dict = {}
	subart_series_df = subart_series_df.sort_values(["level", "condition"])
	for level, level_df in subart_series_df.groupby("level"):
		rt_instance = level_df.instance_number.values[1]
		lt_instance = level_df.instance_number.values[0]
		if np.isnan(rt_instance):
			subart_slice_dict["rt_" + level.replace("/", "_").lower()] = np.nan
		else:
			subart_slice_dict["rt_" + level.replace("/", "_").lower()] = instance_to_position[rt_instance]
		if np.isnan(lt_instance):
			subart_slice_dict["lt_" + level.replace("/", "_").lower()] = np.nan
		else:
			subart_slice_dict["lt_" + level.replace("/", "_").lower()] = instance_to_position[lt_instance]
		subart_coord_dict["rt_" + level.replace("/", "_").lower()] = (level_df.x.values[1], level_df.y.values[1])
		subart_coord_dict["lt_" + level.replace("/", "_").lower()] = (level_df.x.values[0], level_df.y.values[0])
	max_dist = series_df.ImagePositionPatient2.max() - series_df.ImagePositionPatient2.min()
	assert max_dist > 0
	for c in col_names:
		# no rescale
		series_df[c + "_no_rescale"] = series_df.ImagePositionPatient2 - subart_slice_dict[c]
		# add in rescaled coordinates of the subart
		series_df[c + "_subart_coord_x"] = subart_coord_dict[c][0] / series_df.cols
		series_df[c + "_subart_coord_y"] = subart_coord_dict[c][1] / series_df.rows
	# Change coords to -1 if they are not on the right slice or adjacent slices (+/- 1)
	series_df = series_df.reset_index(drop=True)
	for c in col_names:
		correct_slice = np.where(series_df[c + "_no_rescale"] == 0)[0]
		if len(correct_slice) == 0:
			incorrect_slices = list(range(len(series_df)))
		else:
			correct_slice = correct_slice[0]
			correct_slice = np.arange(correct_slice - 1, correct_slice + 2)
			incorrect_slices = list(set(range(len(series_df))) - set(correct_slice))
		series_df.loc[incorrect_slices, f"{c}_subart_coord_x"] = -1
		series_df.loc[incorrect_slices, f"{c}_subart_coord_y"] = -1
	for c in series_df.columns:
		if "_no_rescale" in c:
			series_df[c] = series_df[c].fillna(-88888)
		elif "_coord_" in c:
			series_df[c] = series_df[c].fillna(-1)
	new_df_list.append(series_df)
	break

new_df = pd.concat(new_df_list)

folds_df = pd.read_csv("../../data/folds_cv5.csv")
new_df = new_df.merge(folds_df, on="study_id")

pngfile = []
for row in new_df.itertuples():
	pngfile.append(f"{row.study_id}/{row.series_id}/IM{instance_to_position_index_dict[str(row.series_id) + '_' + str(row.instance_number)]:06d}_INST{row.instance_number:06d}.png")

new_df["pngfile"] = pngfile

new_df.to_csv("../../data/train_subart_distance_each_level_with_subart_coords_and_ignore_imputed.csv", index=False)
