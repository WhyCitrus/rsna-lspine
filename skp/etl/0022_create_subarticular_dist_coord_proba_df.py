import numpy as np
import pandas as pd

from tqdm import tqdm


meta_df = pd.read_csv("../../data/dicom_metadata.csv")
desc_df = pd.read_csv("../../data/train_series_descriptions.csv")
coords_df = pd.read_csv("../../data/train_label_coordinates.csv")

ax_t2_series = desc_df.loc[desc_df.series_description == "Axial T2", "series_id"].tolist()

meta_df = meta_df.loc[meta_df.series_id.isin(ax_t2_series)]
coords_df = coords_df.loc[coords_df.series_id.isin(ax_t2_series)]
levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
dist_cols = [f"rt_{lvl}" for lvl in levels] + [f"lt_{lvl}" for lvl in levels]

coords_df["side_level"] = coords_df.condition.apply(lambda x: x.split()[0][0].lower()) + "t_" + coords_df.level.apply(lambda x: x.replace("/", "_").lower())

df_list = []
for series_id, series_df in tqdm(meta_df.groupby("series_id"), total=len(meta_df.series_id.unique())):
	series_df = series_df.sort_values("ImagePositionPatient2", ascending=True)
	series_df["position_index"] = np.arange(len(series_df))
	tmp_coords_df = coords_df.loc[coords_df.series_id == series_id]
	instance_number_to_position = {row.instance_number: row.ImagePositionPatient2 for row in series_df.itertuples()}
	x_coords = {row.side_level: row.x for row in tmp_coords_df.itertuples()}
	y_coords = {row.side_level: row.y for row in tmp_coords_df.itertuples()}
	for each_dist_col in dist_cols:
		if each_dist_col not in tmp_coords_df.side_level.tolist():
			series_df[each_dist_col + "_no_rescale"] = -88888
			series_df[each_dist_col + "_normalized"] = -88888
		else:
			tmp_coords_row = tmp_coords_df.loc[tmp_coords_df.side_level == each_dist_col]
			assert len(tmp_coords_row) == 1
			series_df[each_dist_col + "_no_rescale"] = series_df.ImagePositionPatient2.values - instance_number_to_position[tmp_coords_row.instance_number.values[0]]
			series_df[each_dist_col + "_normalized"] = series_df[each_dist_col + "_no_rescale"].values / (series_df.ImagePositionPatient2.max() - series_df.ImagePositionPatient2.min())
	for each_dist_col in dist_cols:
		zero_dist = series_df[each_dist_col + "_no_rescale"].values
		zero_dist = np.where(zero_dist == 0)[0]
		tmp_proba_score = np.zeros((len(series_df, )))
		if len(zero_dist) > 0:
			zero_dist = zero_dist[0]
			tmp_proba_score[zero_dist] = 1
			tmp_x, tmp_y = -series_df.cols.values, -series_df.rows.values
			tmp_x[zero_dist] = x_coords[each_dist_col]
			tmp_y[zero_dist] = y_coords[each_dist_col]
			if zero_dist - 1 >= 0:
				tmp_proba_score[zero_dist - 1] = 0.5
				tmp_x[zero_dist - 1] = x_coords.get(each_dist_col, tmp_x[0])
				tmp_y[zero_dist - 1] = y_coords.get(each_dist_col, tmp_y[0])
			if zero_dist + 1 < len(series_df):
				tmp_proba_score[zero_dist + 1] = 0.5
				tmp_x[zero_dist + 1] = x_coords.get(each_dist_col, tmp_x[0])
				tmp_y[zero_dist + 1] = y_coords.get(each_dist_col, tmp_y[0])
			series_df[each_dist_col + "_proba_score"] = tmp_proba_score
			series_df[each_dist_col + "_subarticular_coord_x"] = tmp_x / series_df.cols.values
			series_df[each_dist_col + "_subarticular_coord_y"] = tmp_y / series_df.rows.values
			series_df.loc[series_df[each_dist_col + "_subarticular_coord_x"] == -1, each_dist_col + "_subarticular_coord_x"] = -88888
			series_df.loc[series_df[each_dist_col + "_subarticular_coord_y"] == -1, each_dist_col + "_subarticular_coord_y"] = -88888
		else:
			series_df[each_dist_col + "_proba_score"] = tmp_proba_score
			series_df[each_dist_col + "_subarticular_coord_x"] = -88888
			series_df[each_dist_col + "_subarticular_coord_y"] = -88888
	rt_coord_x_cols = [c for c in series_df.columns if "rt_" in c and c.endswith("_x")]
	lt_coord_x_cols = [c for c in series_df.columns if "lt_" in c and c.endswith("_x")]
	rt_coord_y_cols = [c for c in series_df.columns if "rt_" in c and c.endswith("_y")]
	lt_coord_y_cols = [c for c in series_df.columns if "lt_" in c and c.endswith("_y")]
	series_df["rt_subarticular_coord_x"] = series_df[rt_coord_x_cols].max(1)
	series_df["lt_subarticular_coord_x"] = series_df[lt_coord_x_cols].max(1)
	series_df["rt_subarticular_coord_y"] = series_df[rt_coord_y_cols].max(1)
	series_df["lt_subarticular_coord_y"] = series_df[lt_coord_y_cols].max(1)
	# Take the minimum absolute normalized distance from any level 
	# Idea is that we should sample slices closer to a level more than those further away
	# Since in the end it's not really the exact distance we care about but picking the optimal slice
	series_df["min_normalized_dist"] = series_df[[c for c in series_df.columns if "_normalized" in c]].abs().min(1)
	series_df["inverse_dist_weight"] = 1 / (series_df.min_normalized_dist.values + 0.1)
	series_df["sampling_weight"] = series_df.inverse_dist_weight / series_df.inverse_dist_weight.sum()
	df_list.append(series_df)

df = pd.concat(df_list)
folds_df = pd.read_csv("../../data/folds_cv5.csv")
df = df.merge(folds_df, on="study_id")
df["pngfile"] = df.study_id.astype("str") + "/" + df.series_id.astype("str") + "/" + df.position_index.apply(lambda x: f"IM{x:06d}") + df.instance_number.apply(lambda x: f"_INST{x:06d}.png")
df.to_csv("../../data/train_subarticular_dist_coord_proba_with_weights.csv", index=False)
