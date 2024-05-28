import numpy as np
import pandas as pd


coords_df = pd.read_csv("../../data/train_label_coordinates.csv")
meta_df = pd.read_csv("../../data/dicom_metadata.csv")

num_slices_per_series = {}
for series_id, series_df in meta_df.groupby("series_id"):
    num_slices_per_series[series_id] = len(series_df)

meta_df["num_slices"] = meta_df.series_id.map(num_slices_per_series)

# To start, just pick all the sagittal series where all foramina and spinal canal slices are available
coords_df = coords_df.loc[coords_df.condition.apply(lambda x: "Foraminal" in x or "Canal" in x)]

study_id_list = []
for study_id, study_df in coords_df.groupby("study_id"):
    if len(study_df) == 15:
        study_id_list.append(study_id)

coords_df = coords_df.loc[coords_df.study_id.isin(study_id_list)].reset_index(drop=True)
# coords_df = coords_df.merge(meta_df, on=["study_id", "series_id", "instance_number"])

# Make a dictionary mapping each study_id to its instance_number for each foramina
# and spinal canal at each level
mapping = {}
levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
keys = [f"lt_{l}_foramen" for l in levels] + [f"rt_{l}_foramen" for l in levels] + [f"{l}_canal" for l in levels]

df_list = []
for study_id, study_df in coords_df.groupby("study_id"):
    study_df = study_df.sort_values("condition", ascending=True)
    tmp_dict = {k: study_df.instance_number.iloc[idx] for idx, k in enumerate(keys)}
    tmp_df = pd.DataFrame(tmp_dict, index=[0])
    tmp_df["study_id"] = study_id
    df_list.append(tmp_df)

slice_df = pd.concat(df_list).reset_index(drop=True)

# Now, we are only interested in the sagittal T1 sequences
series_descs = pd.read_csv("../../data/train_series_descriptions.csv")
series_descs = series_descs.loc[series_descs.series_description == "Sagittal T1"]
slice_df = slice_df.merge(series_descs, on="study_id")
slice_df = slice_df.merge(meta_df, on=["study_id", "series_id"])

for k in keys:
    # Calculate distance from target slice of each slice
    slice_df[f"{k}_rel"] = slice_df[k] - slice_df.instance_number

# For foramen, it is not really possible to differentiate between left and right
# based on a single 2D slice
# Thus, we will just treat them the same and use the slice location to figure out
# laterality
# Then, for the target, we will use the lesser of the 2 values 
for each_level in levels:
    slice_df[f"{each_level}_foramen_rel"] = slice_df[[f"lt_{each_level}_foramen_rel", f"rt_{each_level}_foramen_rel"]].abs().min(1)

for each_level in levels:
    slice_df[f"{each_level}_foramen_rel_target"] = slice_df[f"{each_level}_foramen_rel"] / slice_df.num_slices
    slice_df[f"{each_level}_canal_rel_target"] = slice_df[f"{each_level}_canal_rel"].abs() / slice_df.num_slices

df_list = []
for study_id, study_df in slice_df.groupby("study_id"):
    study_df = study_df.sort_values("ImagePositionPatient0", ascending=True)
    positions = study_df.ImagePositionPatient0.values
    position_diffs = positions[:-1] - positions[1:]
    slice_spacing = np.abs(np.median(position_diffs))
    study_df["slice_spacing"] = slice_spacing
    df_list.append(study_df)

slice_df = pd.concat(df_list)

for each_level in levels:
    slice_df[f"{each_level}_foramen_mm"] = slice_df[f"{each_level}_foramen_rel"] * slice_df.slice_spacing
    slice_df[f"{each_level}_canal_mm"] = slice_df[f"{each_level}_canal_rel"].abs() * slice_df.slice_spacing

# slice_df[["instance_number", "lt_l1_l2_foramen", "rt_l1_l2_foramen", "lt_l1_l2_foramen_rel", "rt_l1_l2_foramen_rel", "l1_l2_foramen_rel", "slice_spacing", "l1_l2_foramen_mm"]]

for each_level in levels:
    slice_df[f"{each_level}_canal_cls"] = (slice_df.instance_number == slice_df[f"{each_level}_canal"]).astype("int")
    slice_df[f"{each_level}_foramen_cls"] = (slice_df.instance_number == slice_df[f"lt_{each_level}_foramen"]).astype("int") + (slice_df.instance_number == slice_df[f"rt_{each_level}_foramen"]).astype("int")

for c in slice_df.columns:
    if "_cls" not in c:
        continue
    slice_df[c] = (slice_df[c] > 0).astype("int")

folds_df = pd.read_csv("../../data/folds_cv5.csv")
slice_df = slice_df.merge(folds_df, on="study_id")
slice_df["filepath"] = slice_df.study_id.astype("str") + "/" + slice_df.series_id.astype("str") + "/" + slice_df.instance_number.apply(lambda x: f"IM{x:06d}.png")

slice_df.to_csv("../../data/train_predict_which_sagittal_slice_foramina_canal_2d.csv", index=False)



