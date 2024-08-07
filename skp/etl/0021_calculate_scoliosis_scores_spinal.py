import numpy as np
import pandas as pd


df = pd.read_csv("../../data/train_label_coordinates.csv")
df = df.loc[df.condition.apply(lambda x: "Spinal" in x)]

levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
dist_cols = [f"{lvl}_no_rescale" for lvl in levels]

scoliosis_scores = {}
for series_id, series_df in df.groupby("series_id"):
	if len(series_df.level.unique()) != 5:
		continue
	scoliosis_scores[series_id] = np.std(series_df.instance_number.values)

df["sampling_weight"] = df.series_id.map(scoliosis_scores) + 1
np.percentile(df.sampling_weight.values[~df.sampling_weight.isna()], [0, 5, 10, 15, 20, 25, 50, 75, 80, 85, 90, 95, 100])
df.sampling_weight.value_counts()

df = df[["study_id", "series_id", "sampling_weight"]].drop_duplicates()
df.to_csv("../../data/train_spinal_with_scoliosis_sampling_weights.csv", index=False)

orig_df = pd.read_csv("../../data/train_spinal_dist_each_level_with_coords_and_ignore_upsample_hard_cases.csv")
orig_df = orig_df.drop_duplicates()

df_list = []
for series_id, series_df in orig_df.groupby("series_id"):
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

orig_df = pd.concat(df_list)

orig_df = orig_df.merge(df, on=["study_id", "series_id"])
orig_df = orig_df.loc[~orig_df.sampling_weight.isna()]
orig_df.to_csv("../../data/train_spinal_dist_coord_proba_sampling_weight.csv", index=False)
