import pandas as pd

from utils import create_double_cv


meta_df = pd.read_csv("../../data/dicom_metadata.csv")
coords_df = pd.read_csv("../../data/train_label_coordinates.csv")
series_descs = pd.read_csv("../../data/train_series_descriptions.csv")

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

# Need number of slices per series
num_slices_per_series = {}
for series_id, series_df in meta_df.groupby("series_id"):
    if series_id not in foramina_df.series_id.tolist():
        continue
    num_slices_per_series[series_id] = len(series_df)

foramina_df["rel_z"] = [(row.instance_number - 1) / (num_slices_per_series[row.series_id] - 1) for row in foramina_df.itertuples()]

new_df_list = []
col_names = [
    "lt_foramen_l1_l2_x", "lt_foramen_l2_l3_x", "lt_foramen_l3_l4_x", "lt_foramen_l4_l5_x", "lt_foramen_l5_s1_x",
    "rt_foramen_l1_l2_x", "rt_foramen_l2_l3_x", "rt_foramen_l3_l4_x", "rt_foramen_l4_l5_x", "rt_foramen_l5_s1_x",
    "lt_foramen_l1_l2_y", "lt_foramen_l2_l3_y", "lt_foramen_l3_l4_y", "lt_foramen_l4_l5_y", "lt_foramen_l5_s1_y",
    "rt_foramen_l1_l2_y", "rt_foramen_l2_l3_y", "rt_foramen_l3_l4_y", "rt_foramen_l4_l5_y", "rt_foramen_l5_s1_y",
    "lt_foramen_l1_l2_z", "lt_foramen_l2_l3_z", "lt_foramen_l3_l4_z", "lt_foramen_l4_l5_z", "lt_foramen_l5_s1_z",
    "rt_foramen_l1_l2_z", "rt_foramen_l2_l3_z", "rt_foramen_l3_l4_z", "rt_foramen_l4_l5_z", "rt_foramen_l5_s1_z"
]
for series_id, series_df in foramina_df.groupby("series_id"):
    series_df = series_df.sort_values(["condition", "level"])
    tmp_dict = {"study_id": series_df.study_id.iloc[0], "series_id": series_id}
    for each_col, each_coord in zip(col_names, series_df["rel_x"].tolist() + series_df["rel_y"].tolist() + series_df["rel_z"].tolist()):
        tmp_dict[each_col] = each_coord
    tmp_df = pd.DataFrame(tmp_dict, index=[0])
    new_df_list.append(tmp_df)

new_df = pd.concat(new_df_list).reset_index(drop=True)
new_df["series_folder"] = new_df.study_id.astype("str") + "/" + new_df.series_id.astype("str")
new_df = create_double_cv(new_df, "study_id", 5, 5)
new_df.to_csv("../../data/train_to_test_predicting_sagittal_foramina_coords_kfold.csv", index=False)
