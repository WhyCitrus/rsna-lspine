import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm


df = pd.read_csv("../../data/train_label_coordinates.csv")
# Get subarticular coordinates only
df = df.loc[df.condition.apply(lambda x: "Subarticular" in x)]
# Reformat so that there is only one row per instance number (i.e. slice)
# and right/left subarticular are 2 separate columns
rt_df = df.loc[df.condition.apply(lambda x: "Right" in x)]
lt_df = df.loc[df.condition.apply(lambda x: "Left" in x)]
rt_df["rt_subarticular"] = 1
lt_df["lt_subarticular"] = 1
df = rt_df.merge(lt_df, on=["study_id", "series_id", "instance_number"], suffixes=["_rt", "_lt"], how="outer")

# Get ImagePositionPatient2 data from metadata
meta_df = pd.read_csv("../../data/dicom_metadata.csv")
meta_df = meta_df.loc[meta_df.series_id.isin(df.series_id.tolist())]

# Merge
df = df.merge(meta_df, on=["study_id", "series_id", "instance_number"], how="right")

levels = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
df_list = []
for each_series, series_df in df.groupby("series_id"):
    series_df = series_df.sort_values("ImagePositionPatient2", ascending=False).reset_index(drop=True)
    levels_dict = {each_level: {} for each_level in levels}
    # Use the subarticular slice to define the levels
    # For example, the slice at level L1/L2 is the LAST slice of L1 and everything above is also L1
    # Sometimes left and right are not on the same slice, so use the LOWER of the 2
    for each_level in levels:
        for row_idx, row in series_df.iterrows():
            if row.level_rt == each_level:
                levels_dict[each_level]["rt"] = row_idx
            if row.level_lt == each_level:
                levels_dict[each_level]["lt"] = row_idx
    # If any of the levels are missing one or both sides
    # Skip for simplicity
    skip = False
    for k, v in levels_dict.items():
        if len(v) < 2:
            skip = True
    if skip:
        continue
    for k, v in levels_dict.items():
        levels_dict[k] = max(v["rt"], v["lt"])
    levels_labels = {}
    for k, v in levels_dict.items():
        tmp_label = np.zeros(len(series_df))
        tmp_label[:v + 1] = 1
        levels_labels[k[:2]] = tmp_label
    for k, v in levels_labels.items():
        series_df[k] = v
    abbrev_levels = ["L5", "L4", "L3", "L2", "L1"]
    for ii in range(len(abbrev_levels) - 1):
        series_df[abbrev_levels[ii]] -= series_df[abbrev_levels[ii + 1]]
    df_list.append(series_df)

df = pd.concat(df_list)

# Not exactly sure why 2 of them have values of -1 for L3 but we will just exclude them
for lvl in abbrev_levels:
    df = df.loc[df[lvl] != -1]

df = df.reset_index(drop=True)

# Add an S1 level so we can just take argmax over L1-S1 classes to assign the appropriate level
df["S1"] = (df[["L1", "L2", "L3", "L4", "L5"]].sum(1) == 0).astype("int")

df["rt_subarticular"] = df.rt_subarticular.fillna(0)
df["lt_subarticular"] = df.lt_subarticular.fillna(0)
df["filepath"] = df.study_id.astype("str") + "/" + df.series_id.astype("str") + "/" + df.instance_number.apply(lambda x: f"IM{x:06d}.png")

folds_df = pd.read_csv("../../data/folds_cv5.csv")
folds_dict = {row.study_id: row.fold for row in folds_df.itertuples()}

annotations = []
for row in df.itertuples():
    rt_coords = [] if np.isnan(row.x_rt) or np.isnan(row.y_rt) else np.asarray([row.x_rt, row.y_rt])
    lt_coords = [] if np.isnan(row.x_lt) or np.isnan(row.y_lt) else np.asarray([row.x_lt, row.y_lt])
    if len(rt_coords) == 0:
        rt_coords = np.zeros((0,))
    if len(lt_coords) == 0:
        lt_coords = np.zeros((0,))    
    tmp_ann = {
        "filepath": row.filepath,
        "rt_coords": rt_coords, 
        "lt_coords": lt_coords,
        "labels": np.asarray([row.rt_subarticular, row.lt_subarticular, row.L1, row.L2, row.L3, row.L4, row.L5, row.S1]),
        "fold": folds_dict[row.study_id]
    }
    annotations.append(tmp_ann)

with open("../../data/train_subarticular_seg_cls.pkl", "wb") as f:
    pickle.dump(annotations, f)
