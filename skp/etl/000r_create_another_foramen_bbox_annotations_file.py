import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm


df = pd.read_csv("../../data/train_label_coordinates.csv")
meta_df = pd.read_csv("../../data/dicom_metadata.csv")
df = df.merge(meta_df, on=["study_id", "series_id", "instance_number"])
df = df.loc[df.condition.apply(lambda x: "Foraminal" in x)]

# What is the largest distance (instance number) between levels? 
distance_dict = {}
for series_id, series_df in df.groupby("series_id"):
    rt_df = series_df.loc[series_df.condition.apply(lambda x: "Right" in x)]
    lt_df = series_df.loc[series_df.condition.apply(lambda x: "Left"  in x)]
    distance_dict[f"rt_{series_id}"] = rt_df.instance_number.max() - rt_df.instance_number.min()
    distance_dict[f"lt_{series_id}"] = lt_df.instance_number.max() - lt_df.instance_number.min()

np.unique(list(distance_dict.values()), return_counts=True)
# (array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8., 10., nan]), array([ 817, 1978,  906,  172,   45,   15,    5,    4,    1,    1,   14]))
# vast majority are 2 or under

new_df = []
for series_id, series_df in df.groupby("series_id"):
    rt_df = series_df.loc[series_df.condition.apply(lambda x: "Right" in x)].reset_index(drop=True).copy()
    lt_df = series_df.loc[series_df.condition.apply(lambda x: "Left" in x)].reset_index(drop=True).copy()
    if len(rt_df.level.unique()) == 5:
        if rt_df.instance_number.max() - rt_df.instance_number.min() > 2:
            continue
        if rt_df.instance_number.max() - rt_df.instance_number.min() > 0:
            instance_numbers = rt_df.instance_number.tolist()
            additional_rows = []
            for row_idx, row in rt_df.iterrows():
                if row.instance_number - 1 in instance_numbers:
                    row_copy = rt_df.iloc[row_idx:row_idx + 1].copy()
                    row_copy.instance_number = row.instance_number - 1
                    additional_rows.append(row_copy)
                if row.instance_number + 1 in instance_numbers:
                    row_copy = rt_df.iloc[row_idx:row_idx + 1].copy()
                    row_copy.instance_number = row.instance_number + 1
                    additional_rows.append(row_copy)
            rt_df = pd.concat([rt_df] + additional_rows)
        new_df.append(rt_df)
    if len(lt_df.level.unique()) == 5:
        if lt_df.instance_number.max() - lt_df.instance_number.min() > 2:
            continue
        if lt_df.instance_number.max() - lt_df.instance_number.min() > 0:
            instance_numbers = lt_df.instance_number.tolist()
            additional_rows = []
            for row_idx, row in lt_df.iterrows():
                if row.instance_number - 1 in instance_numbers:
                    row_copy = lt_df.iloc[row_idx:row_idx + 1].copy()
                    row_copy.instance_number = row.instance_number - 1
                    additional_rows.append(row_copy)
                if row.instance_number + 1 in instance_numbers:
                    row_copy = lt_df.iloc[row_idx:row_idx + 1].copy()
                    row_copy.instance_number = row.instance_number + 1
                    additional_rows.append(row_copy)
            lt_df = pd.concat([lt_df] + additional_rows)  
        new_df.append(lt_df)

new_df = pd.concat(new_df)

new_df["w"] = 0.05 * new_df.cols
new_df["h"] = 0.05 * new_df.rows
new_df["x1"] = (new_df.x - new_df.w / 2).astype("int")
new_df["x2"] = (new_df.x + new_df.w / 2).astype("int")
new_df["y1"] = (new_df.y - new_df.h / 2).astype("int")
new_df["y2"] = (new_df.y + new_df.h / 2).astype("int")
new_df["label"] = pd.Categorical(new_df.level).codes
new_df["filepath"] = new_df.study_id.astype("str") + "/" + new_df.series_id.astype("str") + "/" + new_df.instance_number.apply(lambda x: f"IM{x:06d}.png")

folds_df = pd.read_csv("../../data/folds_cv5.csv")
folds_dict = {row.study_id: row.fold for row in folds_df.itertuples()}

annotations = []
for fp, fp_df in tqdm(new_df.groupby("filepath"), total=len(new_df.filepath.unique())):
    if 2 in fp_df.level.value_counts().tolist():
        # should not have two of the same level on 1 image
        continue
    tmp_ann = {
        "filepath": fp,
        "bboxes": fp_df[["x1", "y1", "x2", "y2"]].values, 
        "labels": fp_df["label"].values,
        "fold": folds_dict[fp_df.study_id.iloc[0]]
    }
    annotations.append(tmp_ann)

with open("../../data/train_foramen_bboxes_propagated.pkl", "wb") as f:
    pickle.dump(annotations, f)
