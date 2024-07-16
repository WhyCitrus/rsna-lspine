import numpy as np
import pandas as pd 


df = pd.read_csv("../../data/train_foramen_distance_each_level.csv")

df_list = []
for series_id, series_df in df.groupby("series_id"):
	series_df = series_df.sort_values("ImagePositionPatient0")
	series_df["sag_position"] = np.arange(len(series_df)) / len(series_df)
	df_list.append(series_df)

df = pd.concat(df_list)
df.to_csv("../../data/train_foramen_distance_each_level_with_sag_position.csv", index=False)
