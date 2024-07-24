import pandas as pd


df = pd.read_csv("../../data/train_foramen_dist_each_level_with_coords_and_ignore_upsample_hard_cases.csv")

levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
rt_coord_cols = [f"rt_{lvl}_foramen_coord_x" for lvl in levels] + [f"rt_{lvl}_foramen_coord_y" for lvl in levels]
lt_coord_cols = [_.replace("rt", "lt") for _ in rt_coord_cols]

new_df_list = []
for series_id, series_df in df.groupby("series_id"):
	for rt_col, lt_col in zip(rt_coord_cols, lt_coord_cols):
		col = rt_col.replace("rt_", "")
		series_df[col] = series_df[[rt_col, lt_col]].max(1)
	new_df_list.append(series_df)

df = pd.concat(new_df_list)

df.to_csv("../../data/train_foramen_dist_each_level_with_coords_and_ignore_upsample_hard_cases_side_agnostic.csv", index=False)


df = pd.read_csv("../../data/train_subart_distance_each_level_with_subart_coords_and_ignore_imputed.csv")

levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
rt_coord_x_cols = [f"rt_{lvl}_subart_coord_x" for lvl in levels] 
rt_coord_y_cols = [f"rt_{lvl}_subart_coord_y" for lvl in levels] 
lt_coord_x_cols = [f"lt_{lvl}_subart_coord_x" for lvl in levels] 
lt_coord_y_cols = [f"lt_{lvl}_subart_coord_y" for lvl in levels] 

new_df_list = []
for series_id, series_df in df.groupby("series_id"):
	series_df["rt_subart_coord_x"] = series_df[rt_coord_x_cols].max(1)
	series_df["rt_subart_coord_y"] = series_df[rt_coord_y_cols].max(1)
	series_df["lt_subart_coord_x"] = series_df[lt_coord_x_cols].max(1)
	series_df["lt_subart_coord_y"] = series_df[lt_coord_y_cols].max(1)
	new_df_list.append(series_df)

df = pd.concat(new_df_list)

df.to_csv("../../data/train_subart_distance_each_level_with_subart_coords_and_ignore_imputed_level_agnostic.csv", index=False)
