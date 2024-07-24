import numpy as np
import pandas as pd


coords_df = pd.read_csv("../../data/train_label_coordinates.csv")
meta_df = pd.read_csv("../../data/dicom_metadata.csv")
coords_df = coords_df.loc[coords_df.condition.apply(lambda x: "Subarticular" in x)]
coords_df["condition_level"] = coords_df.condition.apply(lambda x: x.split()[0].lower()) + "_" + coords_df.level.apply(lambda x: x.replace("/", "_").lower())

meta_df = meta_df.loc[meta_df.series_id.isin(coords_df.series_id.tolist())]
instance_to_position_index_dict = {}
for series_id, series_df in meta_df.groupby("series_id"):
    series_df = series_df.sort_values("ImagePositionPatient2", ascending=True)
    series_df["position_index"] = np.arange(len(series_df))
    instance_to_position_index_dict.update({f"{series_id}_{row.instance_number}": row.position_index for row in series_df.itertuples()})

df_list = []
for series_id, series_df in coords_df.groupby("series_id"):
    if len(series_df.condition_level.unique()) == 10:
        df_list.append(series_df)

coords_df = pd.concat(df_list)
coords_df["series_instance"] = coords_df.series_id.astype("str") + "_" + coords_df.instance_number.astype("str")
coords_df["position_index"] = coords_df.series_instance.map(instance_to_position_index_dict)

col_names = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
col_names = [_ + "_position_index" for _ in col_names]

df_list = []
for series_id, series_df in coords_df.groupby("series_id"):
    series_df = series_df.sort_values("condition_level")
    left_df = series_df.iloc[:5]
    right_df = series_df.iloc[5:]
    assert len(left_df) == len(right_df) == 5
    # The lower the position index, the more CAUDAL we are 
    # So let's take the lower of left and right so we use the more inferior boundary
    # Though it won't really matter since we are going to take the midpoint anyways
    level_position_indices = np.stack([left_df.position_index.values, right_df.position_index.values], axis=1).min(1) 
    level_position_indices_dict = {k: v for k, v in zip(col_names, level_position_indices)}
    level_position_indices_dict.update({"series_id": series_id, "study_id": series_df.study_id.iloc[0]})
    tmp_df = pd.DataFrame(level_position_indices_dict, index=[0])
    df_list.append(tmp_df)

level_position_df = pd.concat(df_list)
folds_df = pd.read_csv("../../data/folds_cv5.csv")
level_position_df = level_position_df.merge(folds_df, on="study_id")
level_position_df["features"] = level_position_df.series_id.astype("str") + "_features.npy"

level_position_df.to_csv("../../data/level_position_subarticular_axial_df.csv", index=False)



