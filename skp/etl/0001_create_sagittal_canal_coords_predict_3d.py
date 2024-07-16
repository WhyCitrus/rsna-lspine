import numpy as np
import pandas as pd

from collections import defaultdict


meta_df = pd.read_csv("../../data/dicom_metadata.csv")
coords_df = pd.read_csv("../../data/train_label_coordinates.csv")
series_descs = pd.read_csv("../../data/train_series_descriptions.csv")

spinal_df = coords_df.loc[coords_df.condition.apply(lambda x: "Spinal" in x)]
spinal_df = spinal_df.merge(meta_df, on=["study_id", "series_id", "instance_number"])

missing_rows = []
missing_level = []
inconsistent_y = []
valid_series = []
for series_id, series_df in spinal_df.groupby("series_id"):
    try:
        assert len(series_df) == 5
    except:
        missing_rows.append(series_id)
        continue
    try:
        assert len(series_df.level.unique()) == 5
    except:
        missing_level.append(series_id)
        continue
    series_df = series_df.sort_values("level", ascending=True)
    try:
        for i in range(4):
            assert series_df.y.values[i] < series_df.y.values[i+1]
    except:
        inconsistent_y.append(series_id)
        continue
    valid_series.append(series_id)

print(f"There are {len(valid_series)} / {len(spinal_df.series_id.unique())} valid series.")
spinal_df = spinal_df[spinal_df.series_id.isin(valid_series)]

num_slices_per_series = {series_id: len(series_df) for series_id, series_df in meta_df.groupby("series_id") if series_id in spinal_df.series_id.tolist()}
np.percentile(list(num_slices_per_series.values()), [0, 5, 10, 25, 50, 75, 90, 95, 100])
# array([ 8., 12., 14., 15., 17., 19., 21., 21., 27.])

instance_to_position_index_dict = {}
for series_id, series_df in meta_df.groupby("series_id"):
    if series_id not in spinal_df.series_id.tolist():
        continue
    series_df = series_df.sort_values("ImagePositionPatient0", ascending=True).reset_index(drop=True)
    for row_idx, row in series_df.iterrows():
        instance_to_position_index_dict[f"{series_id}_{row.instance_number}"] = row_idx

spinal_df["series_instance_number"] = spinal_df.series_id.astype("str") + "_" + spinal_df.instance_number.astype("str")
spinal_df["position_index"] = spinal_df.series_instance_number.map(instance_to_position_index_dict)

col_names = [
    "canal_l1_l2_slice", "canal_l1_l2_x", "canal_l1_l2_y",
    "canal_l2_l3_slice", "canal_l2_l3_x", "canal_l2_l3_y",
    "canal_l3_l4_slice", "canal_l3_l4_x", "canal_l3_l4_y",
    "canal_l4_l5_slice", "canal_l4_l5_x", "canal_l4_l5_y",
    "canal_l5_s1_slice", "canal_l5_s1_x", "canal_l5_s1_y"
]

coord_dict = defaultdict(list)
for series_id, series_df in spinal_df.groupby("series_id"):
    series_df = series_df.sort_values("level", ascending=True)
    coordinates = []
    for row_idx, row in series_df.iterrows():
        coordinates.extend([row.position_index / (num_slices_per_series[series_id] - 1), row.x / row.cols, row.y / row.rows])
    coord_dict["study_id"].append(series_df.study_id.iloc[0])
    coord_dict["series_id"].append(series_id)
    for each_col, each_coord in zip(col_names, coordinates):
        coord_dict[each_col].append(each_coord)

coord_df = pd.DataFrame(coord_dict)
# how many coordinates are not within the middle 50% of slices
slice_col_names = [c for c in col_names if "slice" in c]
eccentric_coords = []
for c in slice_col_names:
    eccentric_coords.extend(list(coord_df.series_id[coord_df[c].values < 0.25]))
    eccentric_coords.extend(list(coord_df.series_id[coord_df[c].values > 0.75]))

# only 1 which seems like a mistake at L1/L2, all other levels are 0.5
coord_df.loc[coord_df.series_id == eccentric_coords[0], "canal_l1_l2_slice"] = 0.5


folds_df = pd.read_csv("../../data/folds_cv5.csv")
coord_df = coord_df.merge(folds_df, on="study_id")
coord_df["series_folder"] = coord_df.study_id.astype("str") + "/" + coord_df.series_id.astype("str")

coord_df.to_csv("../../data/train_sagittal_canal_coords_3d_kfold.csv", index=False)
