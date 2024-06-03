import pandas as pd


df = pd.read_csv("../../data/train_label_coordinates.csv")
df = df.loc[df.condition.apply(lambda x: "Subarticular" in x)]

# Are there studies with more than 1 series listed? 
more_than_one_series = []
for study_id, study_df in df.groupby("study_id"):
	if len(study_df.series_id.unique()) > 1:
		more_than_one_series.append(study_id)

# Yes, 335 ... Need to figure out why later

# Only take fully labeled series
df_list = []
for series_id, series_df in df.groupby("series_id"):
	if len(series_df) == 10:
		df_list.append(series_df)

full_df = pd.concat(df_list)

meta_df = pd.read_csv("../../data/dicom_metadata.csv")
full_df = full_df.merge(meta_df, on=["study_id", "series_id", "instance_number"])

full_df["rel_x"] = full_df["x"] / full_df["cols"]
full_df["rel_y"] = full_df["y"] / full_df["rows"]

# Need number of slices per series
num_slices_per_series = {}
for series_id, series_df in meta_df.groupby("series_id"):
    if series_id not in full_df.series_id.tolist():
        continue
    num_slices_per_series[series_id] = len(series_df)

full_df["rel_z"] = [(row.instance_number - 1) / (num_slices_per_series[row.series_id] - 1) for row in full_df.itertuples()]

new_df_list = []
col_names = [
    "lt_subarticular_l1_l2_x", "lt_subarticular_l2_l3_x", "lt_subarticular_l3_l4_x", "lt_subarticular_l4_l5_x", "lt_subarticular_l5_s1_x",
    "rt_subarticular_l1_l2_x", "rt_subarticular_l2_l3_x", "rt_subarticular_l3_l4_x", "rt_subarticular_l4_l5_x", "rt_subarticular_l5_s1_x",
    "lt_subarticular_l1_l2_y", "lt_subarticular_l2_l3_y", "lt_subarticular_l3_l4_y", "lt_subarticular_l4_l5_y", "lt_subarticular_l5_s1_y",
    "rt_subarticular_l1_l2_y", "rt_subarticular_l2_l3_y", "rt_subarticular_l3_l4_y", "rt_subarticular_l4_l5_y", "rt_subarticular_l5_s1_y",
    "lt_subarticular_l1_l2_z", "lt_subarticular_l2_l3_z", "lt_subarticular_l3_l4_z", "lt_subarticular_l4_l5_z", "lt_subarticular_l5_s1_z",
    "rt_subarticular_l1_l2_z", "rt_subarticular_l2_l3_z", "rt_subarticular_l3_l4_z", "rt_subarticular_l4_l5_z", "rt_subarticular_l5_s1_z"
]
for series_id, series_df in full_df.groupby("series_id"):
    series_df = series_df.sort_values(["condition", "level"])
    tmp_dict = {"study_id": series_df.study_id.iloc[0], "series_id": series_id}
    for each_col, each_coord in zip(col_names, series_df["rel_x"].tolist() + series_df["rel_y"].tolist() + series_df["rel_z"].tolist()):
        tmp_dict[each_col] = each_coord
    tmp_df = pd.DataFrame(tmp_dict, index=[0])
    new_df_list.append(tmp_df)

new_df = pd.concat(new_df_list).reset_index(drop=True)
new_df["series_folder"] = new_df.study_id.astype("str") + "/" + new_df.series_id.astype("str")
folds_df = pd.read_csv("../../data/folds_cv5.csv")
new_df = new_df.merge(folds_df, on="study_id")
new_df.to_csv("../../data/train_to_test_predicting_axial_subarticular_coords_kfold.csv", index=False)
