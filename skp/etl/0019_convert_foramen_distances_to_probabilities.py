import numpy as np
import pandas as pd


df = pd.read_csv("../../data/train_foramen_dist_each_level_with_coords_and_ignore_upsample_hard_cases_side_agnostic.csv")
df = df.drop_duplicates().reset_index(drop=True)

levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
dist_cols = [f"rt_{lvl}_no_rescale" for lvl in levels] + [f"lt_{lvl}_no_rescale" for lvl in levels]

df_list = []
for series_id, series_df in df.groupby("series_id"):
	series_df = series_df.reset_index(drop=True)
	for each_dist in dist_cols:
		values = np.zeros((len(series_df), ))
		target_slice = np.where(series_df[each_dist].values == 0)[0][0]
		values[target_slice] = 1
		if target_slice - 1 >= 0:
			values[target_slice - 1] = 0.5
		if target_slice + 1 < len(series_df):
			values[target_slice + 1] = 0.5
		series_df[each_dist.replace("no_rescale", "proba_score")] = values
	df_list.append(series_df)

df = pd.concat(df_list)

exclude = [
	3742728457, 757619082, 364930790, 2410494888, 3156269631, 1085426528, 1647904243, 2135829458, 2626030939, 
	1879696087, 3495818564, 1395773918, 2388577668, 2530679352, 2662989538,
]

error_df = pd.read_csv("../../notebooks/large_foramina_slice_instance_error_dist_coord_seg_v3.csv")
error_df = error_df.loc[~error_df.study_id.isin(exclude)]

difficult_studies = df.loc[df.study_id.isin(error_df.study_id.tolist())]
# upsample
df = pd.concat([df] + [difficult_studies] * 3)

df.to_csv("../../data/train_foramen_dist_each_level_with_coords_upsample_hard_side_agnostic_proba_score.csv", index=False)
