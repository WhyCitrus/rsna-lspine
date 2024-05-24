import pandas as pd


coords_df = pd.read_csv("../../data/train_label_coordinates.csv")
# To start, just pick all the sagittal series where all foramina coordinates are available
foramina_df = coords_df.loc[coords_df.condition.apply(lambda x: "Foraminal" in x)]

series_id_list = []
for series_id, series_df in foramina_df.groupby("series_id"):
    if len(series_df) == 10:
        series_id_list.append(series_id)

foramina_df = foramina_df.loc[foramina_df.series_id.isin(series_id_list)].reset_index(drop=True)
foramina_df = foramina_df.merge(meta_df, on=["study_id", "series_id", "instance_number"])

foramina_df["rel_x"] = foramina_df["x"] / foramina_df["cols"]
foramina_df["rel_y"] = foramina_df["y"] / foramina_df["rows"]

col_names = [
    "l1_l2_x", "l2_l3_x", "l3_l4_x", "l4_l5_x", "l5_s1_x",
    "l1_l2_y", "l2_l3_y", "l3_l4_y", "l4_l5_y", "l5_s1_y"
]

df_list = []
for series_id, series_df in foramina_df.groupby("series_id"):
    series_df = series_df.sort_values(["condition", "level"])
    tmp_df1 = series_df[["study_id", "series_id", "instance_number"]].iloc[:5].drop_duplicates().reset_index(drop=True)
    tmp_df2 = series_df[["study_id", "series_id", "instance_number"]].iloc[5:].drop_duplicates().reset_index(drop=True)
    coords1 = series_df.iloc[:5].x.tolist() + series_df.iloc[:5].y.tolist()
    coords1 = pd.DataFrame(np.repeat(np.expand_dims(np.asarray(coords1), axis=0), len(tmp_df1), axis=0))
    coords2 = series_df.iloc[5:].x.tolist() + series_df.iloc[5:].y.tolist()
    coords2 = pd.DataFrame(np.repeat(np.expand_dims(np.asarray(coords2), axis=0), len(tmp_df2), axis=0))
    coords1.columns = col_names
    coords2.columns = col_names
    tmp_df1 = pd.concat([tmp_df1, coords1], axis=1)
    tmp_df2 = pd.concat([tmp_df2, coords2], axis=1)
    tmp_df = pd.concat([tmp_df1, tmp_df2])
    df_list.append(tmp_df)

df = pd.concat(df_list)