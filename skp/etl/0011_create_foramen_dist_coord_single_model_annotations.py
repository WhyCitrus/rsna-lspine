import numpy as np
import pandas as pd
import pickle


df = pd.read_csv("../../data/train_foramen_distance_each_level_with_foramen_coords_and_ignore.csv")
folds_df = pd.read_csv("../../data/folds_cv5.csv")
study_id_to_fold = {row.study_id: row.fold for row in folds_df.itertuples()}

levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]

coord_cols = [f"rt_{lvl}_foramen_coord_x" for lvl in levels] + [f"lt_{lvl}_foramen_coord_x" for lvl in levels]
coord_cols += [f"rt_{lvl}_foramen_coord_y" for lvl in levels]
coord_cols += [f"lt_{lvl}_foramen_coord_y" for lvl in levels]

annotations = []
for series_id, series_df in df.groupby("series_id"):
	series_df = series_df.sort_values("ImagePositionPatient0")
	coord_labels = []
	for c in coord_cols:
		coord_labels.append(np.unique(series_df[c].values[series_df[c].values != -1])[0])
	ann = {
		"study_id": series_df.study_id.values[0],
		"series_id": series_id,
		"filepaths": series_df.pngfile.values,
		"dist_labels": series_df[[f"rt_{lvl}_no_rescale" for lvl in levels] + [f"lt_{lvl}_no_rescale" for lvl in levels]].values,
		"coord_labels": np.asarray(coord_labels),
		"positions": series_df.ImagePositionPatient0.values,
		"fold": study_id_to_fold[series_df.study_id.values[0]]
	}
	annotations.append(ann)


with open("../../data/train_foramen_dist_coord_single_model_annotations.pkl", "wb") as f:
	pickle.dump(annotations, f)
