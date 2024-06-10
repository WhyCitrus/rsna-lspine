import pandas as pd 


df = pd.read_csv("../../data/train_label_coordinates.csv")
df = df.loc[df.condition.apply(lambda x: "Foraminal" in x)]

df_list = []
for series_id, series_df in df.groupby("series_id"):
	right_df = series_df.loc[series_df.condition.apply(lambda x: "Right" in x)]
	left_df = series_df.loc[series_df.condition.apply(lambda x: "Left" in x)]
	bilat_df = right_df.merge(left_df, on=["study_id", "series_id", "level"], suffixes=["_rt", "_lt"])
	if len(bilat_df.level.unique()) == len(bilat_df) == 5:
		df_list.append(bilat_df)

df = pd.concat(df_list)
df["midline_instance"] = (df.instance_number_rt + df.instance_number_lt) // 2
df["x_mid"] = (df.x_rt + df.x_lt) // 2
df["y_mid"] = (df.y_rt + df.y_lt) // 2