import numpy as np
import pandas as pd


df = pd.read_csv("../../data/train_label_coordinates.csv")
df = df.loc[df.condition.apply(lambda x: "Foraminal" in x)]

# The following studies are incorrectly labeled (e.g., left and right sides are swapped or one side is labeled left and right)
incorrectly_labeled = [
	3742728457, 364930790, 2410494888, 2135829458, 1879696087, 3156269631, 1085426528, 3781188430, 3495818564,
	1395773918, 757619082, 2626030939, 2530679352, 2388577668, 1647904243, 796739553, 2662989538
]

df = df.loc[~df.study_id.isin(incorrectly_labeled)]
df["condition_level"] = df.condition.apply(lambda x: x.split()[0].lower()) + "_" + df.level.apply(lambda x: x.replace("/", "_").lower())

scoliosis_scores = {}
for series_id, series_df in df.groupby("series_id"):
	if len(series_df.condition_level.unique()) != 10:
		continue
	rt = series_df.loc[series_df.condition.apply(lambda x: "Right" in x)]
	lt = series_df.loc[series_df.condition.apply(lambda x: "Left" in x)]
	scoliosis_scores[series_id] = (np.std(rt.instance_number.values) + np.std(lt.instance_number.values)) / 2

df["sampling_weight"] = df.series_id.map(scoliosis_scores) + 1
np.percentile(df.sampling_weight.values[~df.sampling_weight.isna()], [0, 5, 10, 15, 20, 25, 50, 75, 80, 85, 90, 95, 100])
df.sampling_weight.value_counts()

df = df[["study_id", "series_id", "sampling_weight"]].drop_duplicates()
df.to_csv("../../data/train_foramina_with_scoliosis_sampling_weights.csv", index=False)

orig_df = pd.read_csv("../../data/train_foramen_dist_each_level_with_coords_upsample_hard_side_agnostic_proba_score.csv")
orig_df = orig_df.drop_duplicates()
orig_df = orig_df.merge(df, on=["study_id", "series_id"])
orig_df = orig_df.loc[~orig_df.sampling_weight.isna()]
orig_df.to_csv("../../data/train_foramen_dist_coord_proba_sampling_weight.csv", index=False)
