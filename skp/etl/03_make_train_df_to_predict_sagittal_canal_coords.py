import pandas as pd

from utils import create_double_cv


meta_df = pd.read_csv("../../data/dicom_metadata.csv")
coords_df = pd.read_csv("../../data/train_label_coordinates.csv")
series_descs = pd.read_csv("../../data/train_series_descriptions.csv")

# To start, just pick all the sagittal series where all foramina coordinates are available
canal_df = coords_df.loc[coords_df.condition.apply(lambda x: "Spinal Canal" in x)]

series_id_list = []
for series_id, series_df in canal_df.groupby("series_id"):
    if len(series_df) == 5:
        series_id_list.append(series_id)

canal_df = canal_df.loc[canal_df.series_id.isin(series_id_list)].reset_index(drop=True)
canal_df = canal_df.merge(meta_df, on=["study_id", "series_id", "instance_number"])

canal_df["rel_x"] = canal_df["x"] / canal_df["cols"]
canal_df["rel_y"] = canal_df["y"] / canal_df["rows"]

# Need number of slices per series
num_slices_per_series = {}
for series_id, series_df in meta_df.groupby("series_id"):
    if series_id not in canal_df.series_id.tolist():
        continue
    num_slices_per_series[series_id] = len(series_df)

canal_df["rel_z"] = [(row.instance_number - 1) / (num_slices_per_series[row.series_id] - 1) for row in canal_df.itertuples()]

new_df_list = []
col_names = [
    "canal_l1_l2_x", "canal_l2_l3_x", "canal_l3_l4_x", "canal_l4_l5_x", "canal_l5_s1_x",
    "canal_l1_l2_y", "canal_l2_l3_y", "canal_l3_l4_y", "canal_l4_l5_y", "canal_l5_s1_y",
    "canal_l1_l2_z", "canal_l2_l3_z", "canal_l3_l4_z", "canal_l4_l5_z", "canal_l5_s1_z"
]
for series_id, series_df in canal_df.groupby("series_id"):
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
new_df.to_csv("../../data/train_to_test_predicting_sagittal_canal_coords_kfold.csv", index=False)
