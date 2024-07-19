import numpy as np
import pandas as pd

from collections import defaultdict


meta_df = pd.read_csv("../../data/dicom_metadata.csv")
coords_df = pd.read_csv("../../data/train_label_coordinates.csv")
series_descs = pd.read_csv("../../data/train_series_descriptions.csv")

subart_df = coords_df.loc[coords_df.condition.apply(lambda x: "Subarticular" in x)]
subart_df = subart_df.merge(meta_df, on=["study_id", "series_id", "instance_number"])

missing_rows = []
missing_condition_level = []
inconsistent_instance = []
inconsistent_x = []
valid_studies = []
multiple_series = []
num_series = []
left_and_right_on_different_series = []
for study_id, study_df in subart_df.groupby("study_id"):
    condition_level = study_df.condition + " " + study_df.level
    if len(study_df.series_id.unique()) > 1:
        multiple_series.append(study_id)
        num_series.append(len(study_df.series_id.unique()))
        for series_id, series_df in study_df.groupby("series_id"):
            # for some reason, some studies have left annotated on one series
            # and right annotated on another series
            if series_df.condition.apply(lambda x: "Left" in x).sum() == 0:
                left_and_right_on_different_series.append(study_id)
                break
            elif series_df.condition.apply(lambda x: "Right" in x).sum() == 0:
                left_and_right_on_different_series.append(study_id)
                break
    try:
        assert len(study_df) == 10
    except:
        missing_rows.append(study_id)
        continue
    try:
        assert len(condition_level.unique()) == 10
    except:
        missing_condition_level.append(study_id)
        continue
    rt = study_df.loc[study_df.condition.apply(lambda x: "Right" in x)]
    lt = study_df.loc[study_df.condition.apply(lambda x: "Left" in x)]
    rt = rt.sort_values("level", ascending=True)
    lt = lt.sort_values("level", ascending=True)
    try:
        for i in range(4):
            assert rt.ImagePositionPatient2.values[i] > rt.ImagePositionPatient2.values[i+1]
            assert lt.ImagePositionPatient2.values[i] > lt.ImagePositionPatient2.values[i+1]
    except:
        inconsistent_instance.append(study_id)
        continue
    try:
        for rx, lx in zip(rt.x, lt.x):
            assert rx < lx
    except:
        inconsistent_x.append(study_id)
        continue
    valid_studies.append(study_id)


print(f"There are {len(valid_studies)} / {len(subart_df.study_id.unique())} valid studies.")
print(f"There are {len(multiple_series)} studies with multiple series.")
print(f"The most number of series in a study is {np.max(num_series)}.")
print(f"There are {len(left_and_right_on_different_series)} studies where right and left are annotated on different series.")

subart_df = subart_df[subart_df.series_id.isin(valid_series)]

num_slices_per_series = {series_id: len(series_df) for series_id, series_df in meta_df.groupby("series_id") if series_id in subart_df.series_id.tolist()}
np.percentile(list(num_slices_per_series.values()), [0, 5, 10, 25, 50, 75, 90, 95, 100])


# which studies are missing subarticular grades?
train_df = pd.read_csv("../../data/train.csv")
# this is likely due to incomplete axial T2 series
# we can then leverage this to train a model that predicts
# which levels are present in an axial T2 stack
subart_cols = [c for c in train_df.columns if "subarticular" in c]
train_subart_df = train_df[["study_id"] + subart_cols]
missing_subart_df = train_subart_df.loc[train_subart_df[subart_cols].isna().sum(1) > 0]
# check to make sure that if one side is missing for a level the contralateral side is also missing
# otherwise this wouldn't make sense
levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
missing_unilateral = []
col_names = [f"subarticular_stenosis_{lvl}" for lvl in levels]
for row_idx, row in missing_subart_df.iterrows():
    for c in col_names:
        if str(row[f"left_{c}"]) == "nan":
            if str(row[f"right_{c}"]) != "nan":
                missing_unilateral.append(row.study_id)
        if str(row[f"right_{c}"]) == "nan":
            if str(row[f"left_{c}"]) != "nan":
                missing_unilateral.append(row.study_id)

missing_unilateral, counts = np.unique(missing_unilateral, return_counts=True)
# 1524089207 missing only left side but images are fine
# these are probably all mistakes



import numpy as np
import os
import pandas as pd
import pickle

from collections import defaultdict


meta_df = pd.read_csv("../../data/dicom_metadata.csv")
coords_df = pd.read_csv("../../data/train_label_coordinates.csv")
series_descs = pd.read_csv("../../data/train_series_descriptions.csv")

subart_df = coords_df.loc[coords_df.condition.apply(lambda x: "Subarticular" in x)]
subart_df = subart_df.merge(meta_df, on=["study_id", "series_id", "instance_number"])

# Let's assume that the instance number (e.g., slice index) annotation correctly identifies the level of interest
# Then, we can use that to demarcate the levels
# Then, we can use that information to generate crops with varying number of levels
# So we can train a classifier to determine which levels are present in a stack of axial images
full_series = []
for series_id, series_df in subart_df.groupby("series_id"):
    if len(series_df) == 10:
        condition_level = series_df.condition + series_df.level
        if len(np.unique(condition_level)) == 10:
            rt = series_df.loc[series_df.condition.apply(lambda x: "Right" in x)]
            lt = series_df.loc[series_df.condition.apply(lambda x: "Left" in x)]
            rt = rt.sort_values("level", ascending=True)
            lt = lt.sort_values("level", ascending=True)
            if np.max(np.abs(rt.instance_number.values - lt.instance_number.values)) <= 3:
                full_series.append(series_id)

subart_df = subart_df.loc[subart_df.series_id.isin(full_series)]

instance_to_position_index_dict = {}
for series_id, series_df in meta_df.groupby("series_id"):
    if series_id not in subart_df.series_id.tolist():
        continue
    series_df = series_df.sort_values("ImagePositionPatient2", ascending=True).reset_index(drop=True)
    for row_idx, row in series_df.iterrows():
        instance_to_position_index_dict[f"{series_id}_{row.instance_number}"] = row_idx

subart_df["series_instance_number"] = subart_df.series_id.astype("str") + "_" + subart_df.instance_number.astype("str")
subart_df["position_index"] = subart_df.series_instance_number.map(instance_to_position_index_dict)

num_slices_per_series = {series_id: len(series_df) for series_id, series_df in meta_df.groupby("series_id") if series_id in subart_df.series_id.tolist()}
np.percentile(list(num_slices_per_series.values()), [0, 5, 10, 25, 50, 75, 90, 95, 100])

folds_df = pd.read_csv("../../data/folds_cv5.csv")
study_folds = {row.study_id: row.fold for row in folds_df.itertuples()}

annotations = []
for series_id, series_df in subart_df.groupby("series_id"):
    ann = {}
    ann["series_id"] = series_id
    ann["series_folder"] = os.path.join(str(series_df.study_id.values[0]), str(series_id))
    series_df = series_df.sort_values("ImagePositionPatient2", ascending=True)
    series_levels_mark = {
        level: (instance_to_position_index_dict[f"{series_id}_{level_df.instance_number.min()}"], 
                instance_to_position_index_dict[f"{series_id}_{level_df.instance_number.max()}"]) 
        for level, level_df in series_df.groupby("level")
    }
    # Need to find the midpoints 
    # Otherwise the end of the stack will barely include the desired disc level
    # Need some buffer
    midpoints = {}
    midpoints["L1"] = num_slices_per_series[series_id] - 1
    midpoints["L2"] = (min(series_levels_mark["L1/L2"]) + max(series_levels_mark["L2/L3"])) // 2
    midpoints["L3"] = (min(series_levels_mark["L2/L3"]) + max(series_levels_mark["L3/L4"])) // 2
    midpoints["L4"] = (min(series_levels_mark["L3/L4"]) + max(series_levels_mark["L4/L5"])) // 2
    midpoints["L5"] = (min(series_levels_mark["L4/L5"]) + max(series_levels_mark["L5/S1"])) // 2
    midpoints["S1"] = 0
    ann["level_ranges"] = {
        "L1/L2": (midpoints["L1"], midpoints["L2"]),
        "L2/L3": (midpoints["L2"], midpoints["L3"]),
        "L3/L4": (midpoints["L3"], midpoints["L4"]),
        "L4/L5": (midpoints["L4"], midpoints["L5"]),
        "L5/S1": (midpoints["L5"], midpoints["S1"])
    }
    ann["coords"] = {
        f"{cond.split()[0][:1]}_{lvl}": [instance_to_position_index_dict[f"{series_id}_{slice_i}"], xi / series_df.cols.values[0], yi / series_df.rows.values[0]] 
        for cond, lvl, slice_i, xi, yi in zip(series_df.condition, series_df.level, series_df.instance_number, series_df.x, series_df.y)
    }
    ann["num_slices"] = num_slices_per_series[series_id]
    ann["fold"] = study_folds[series_df.study_id.iloc[0]]
    annotations.append(ann)

with open("../../data/train_subarticular_levels_and_coords_3d.pkl", "wb") as f:
    pickle.dump(annotations, f)















