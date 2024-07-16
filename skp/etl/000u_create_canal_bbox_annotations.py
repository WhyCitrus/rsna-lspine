import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm


df = pd.read_csv("../../data/train_label_coordinates.csv")
meta_df = pd.read_csv("../../data/dicom_metadata.csv")
df = df.merge(meta_df, on=["study_id", "series_id", "instance_number"])
df = df.loc[df.condition.apply(lambda x: "Canal" in x)]

# What is the largest distance (instance number) between levels? 
distance_dict = {}
for series_id, series_df in df.groupby("series_id"):
    distance_dict[series_id] = series_df.instance_number.max() - series_df.instance_number.min()

np.unique(list(distance_dict.values()), return_counts=True)
# (array([0, 1, 2, 3, 4, 5, 6, 8, 9]), array([1541,  315,   77,   22,   11,    5,    1,    1,    1]))
# vast majority are 2 or under

new_df = []
for series_id, series_df in df.groupby("series_id"):
    series_df_copy = series_df.reset_index(drop=True).copy()
    if len(series_df_copy.level.unique()) == 5:
        if series_df_copy.instance_number.max() - series_df_copy.instance_number.min() > 2:
            continue
        if series_df_copy.instance_number.max() - series_df_copy.instance_number.min() > 0:
            instance_numbers = series_df_copy.instance_number.tolist()
            additional_rows = []
            for row_idx, row in series_df_copy.iterrows():
                if row.instance_number - 1 in instance_numbers:
                    row_copy = series_df_copy.iloc[row_idx:row_idx + 1].copy()
                    row_copy.instance_number = row.instance_number - 1
                    additional_rows.append(row_copy)
                if row.instance_number + 1 in instance_numbers:
                    row_copy = series_df_copy.iloc[row_idx:row_idx + 1].copy()
                    row_copy.instance_number = row.instance_number + 1
                    additional_rows.append(row_copy)
            series_df_copy = pd.concat([series_df_copy] + additional_rows)
        new_df.append(series_df_copy)

new_df = pd.concat(new_df)

new_df["w"] = 0.075 * new_df.cols
new_df["h"] = 0.075 * new_df.rows
new_df["x1"] = (new_df.x - new_df.w / 2).astype("int")
new_df["x2"] = (new_df.x + new_df.w / 2).astype("int")
new_df["y1"] = (new_df.y - new_df.h / 2).astype("int")
new_df["y2"] = (new_df.y + new_df.h / 2).astype("int")
new_df["label"] = pd.Categorical(new_df.level).codes
new_df["filepath"] = new_df.study_id.astype("str") + "/" + new_df.series_id.astype("str") + "/" + new_df.instance_number.apply(lambda x: f"IM{x:06d}.png")

new_df.x1 = [max(0, _) for _ in new_df.x1]
new_df.y1 = [max(0, _) for _ in new_df.y1]
new_df.x2 = [min(i-1, j) for i, j in zip(new_df.cols, new_df.x2)]
new_df.y2 = [min(i-1, j) for i, j in zip(new_df.rows, new_df.y2)]

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

with open("../../data/train_canal_bboxes_propagated.pkl", "wb") as f:
    pickle.dump(annotations, f)
