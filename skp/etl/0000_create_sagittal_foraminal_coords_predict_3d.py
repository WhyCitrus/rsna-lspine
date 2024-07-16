import numpy as np
import pandas as pd

from collections import defaultdict


meta_df = pd.read_csv("../../data/dicom_metadata.csv")
coords_df = pd.read_csv("../../data/train_label_coordinates.csv")
series_descs = pd.read_csv("../../data/train_series_descriptions.csv")

foramina_df = coords_df.loc[coords_df.condition.apply(lambda x: "Foraminal" in x)]
foramina_df = foramina_df.merge(meta_df, on=["study_id", "series_id", "instance_number"])

missing_rows = []
missing_condition_level = []
inconsistent_y = []
inconsistent_instance = []
valid_series = []
for series_id, series_df in foramina_df.groupby("series_id"):
    condition_level = series_df.condition + " " + series_df.level
    try:
        assert len(series_df) == 10
    except:
        missing_rows.append(series_id)
        continue
    try:
        assert len(condition_level.unique()) == 10
    except:
        missing_condition_level.append(series_id)
        continue
    rt = series_df.loc[series_df.condition.apply(lambda x: "Right" in x)]
    lt = series_df.loc[series_df.condition.apply(lambda x: "Left" in x)]
    rt = rt.sort_values("level", ascending=True)
    lt = lt.sort_values("level", ascending=True)
    try:
        for i in range(4):
            assert rt.y.values[i] < rt.y.values[i+1]
            assert lt.y.values[i] < lt.y.values[i+1]
    except:
        inconsistent_y.append(series_id)
        continue
    try:
        for rt_inst, lt_inst in zip(rt.ImagePositionPatient0, lt.ImagePositionPatient0):
            assert rt_inst < lt_inst
    except:
        inconsistent_instance.append(series_id)
        continue
    valid_series.append(series_id)

print(f"There are {len(valid_series)} / {len(foramina_df.series_id.unique())} valid series.")
foramina_df = foramina_df[foramina_df.series_id.isin(valid_series)]

num_slices_per_series = {series_id: len(series_df) for series_id, series_df in meta_df.groupby("series_id") if series_id in foramina_df.series_id.tolist()}
np.percentile(list(num_slices_per_series.values()), [0, 5, 10, 25, 50, 75, 90, 95, 100])
# array([10., 12., 14., 15., 17., 19., 21., 21., 38.])

instance_to_position_index_dict = {}
for series_id, series_df in meta_df.groupby("series_id"):
    if series_id not in foramina_df.series_id.tolist():
        continue
    series_df = series_df.sort_values("ImagePositionPatient0", ascending=True).reset_index(drop=True)
    for row_idx, row in series_df.iterrows():
        instance_to_position_index_dict[f"{series_id}_{row.instance_number}"] = row_idx

foramina_df["series_instance_number"] = foramina_df.series_id.astype("str") + "_" + foramina_df.instance_number.astype("str")
foramina_df["position_index"] = foramina_df.series_instance_number.map(instance_to_position_index_dict)

col_names = [
    "rt_foramen_l1_l2_slice", "rt_foramen_l1_l2_x", "rt_foramen_l1_l2_y",
    "rt_foramen_l2_l3_slice", "rt_foramen_l2_l3_x", "rt_foramen_l2_l3_y",
    "rt_foramen_l3_l4_slice", "rt_foramen_l3_l4_x", "rt_foramen_l3_l4_y",
    "rt_foramen_l4_l5_slice", "rt_foramen_l4_l5_x", "rt_foramen_l4_l5_y",
    "rt_foramen_l5_s1_slice", "rt_foramen_l5_s1_x", "rt_foramen_l5_s1_y",
    "lt_foramen_l1_l2_slice", "lt_foramen_l1_l2_x", "lt_foramen_l1_l2_y",
    "lt_foramen_l2_l3_slice", "lt_foramen_l2_l3_x", "lt_foramen_l2_l3_y",
    "lt_foramen_l3_l4_slice", "lt_foramen_l3_l4_x", "lt_foramen_l3_l4_y",
    "lt_foramen_l4_l5_slice", "lt_foramen_l4_l5_x", "lt_foramen_l4_l5_y",
    "lt_foramen_l5_s1_slice", "lt_foramen_l5_s1_x", "lt_foramen_l5_s1_y"
]

coord_dict = defaultdict(list)
for series_id, series_df in foramina_df.groupby("series_id"):
    series_df = series_df.sort_values(["condition", "level"], ascending=[False, True])
    coordinates = []
    for row_idx, row in series_df.iterrows():
        coordinates.extend([row.position_index / (num_slices_per_series[series_id] - 1), row.x / row.cols, row.y / row.rows])
    coord_dict["study_id"].append(series_df.study_id.iloc[0])
    coord_dict["series_id"].append(series_id)
    for each_col, each_coord in zip(col_names, coordinates):
        coord_dict[each_col].append(each_coord)

coord_df = pd.DataFrame(coord_dict)
folds_df = pd.read_csv("../../data/folds_cv5.csv")
coord_df = coord_df.merge(folds_df, on="study_id")
coord_df["series_folder"] = coord_df.study_id.astype("str") + "/" + coord_df.series_id.astype("str")

coord_df.to_csv("../../data/train_sagittal_foramina_coords_3d_kfold.csv", index=False)
