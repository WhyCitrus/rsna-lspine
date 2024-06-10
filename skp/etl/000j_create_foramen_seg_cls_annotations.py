import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm


df = pd.read_csv("../../data/train_label_coordinates.csv")
meta_df = pd.read_csv("../../data/dicom_metadata.csv")
series_descriptions = pd.read_csv("../../data/train_series_descriptions.csv")
sagittal_t1_series = series_descriptions.loc[series_descriptions.series_description == "Sagittal T1", "series_id"].tolist()
meta_df = meta_df.loc[meta_df.series_id.isin(sagittal_t1_series)]
meta_df["filepath"] = meta_df.study_id.astype("str") + "/" + meta_df.series_id.astype("str") + "/" + meta_df.instance_number.apply(lambda x: f"IM{x:06d}.png")
df = df.merge(meta_df, on=["study_id", "series_id", "instance_number"])
df = df.loc[df.condition.apply(lambda x: "Foraminal" in x)]

df["label"] = pd.Categorical(df.level).codes
negative_foramen_df = meta_df.loc[~meta_df.filepath.isin(df.filepath)]

folds_df = pd.read_csv("../../data/folds_cv5.csv")
folds_dict = {row.study_id: row.fold for row in folds_df.itertuples()}

annotations = []
for fp, fp_df in tqdm(df.groupby("filepath"), total=len(df.filepath.unique())):
    if fp_df.study_id.iloc[0] in [1395773918, 1879696087, 2388577668, 2662989538, 3495818564]:
        # these have both sides foramina on the same slice which makes no sense
        fp_df = fp_df.iloc[:len(fp_df) // 2]
    if fp_df.study_id.iloc[0] in [2410494888, 1513597136, 2530679352]:
        # same
        fp_df = fp_df.iloc[:len(fp_df) // 2 + 1]
    if fp_df.study_id.iloc[0] in [3156269631] and fp_df.instance_number.iloc[0] == 11:
        # same
        fp_df = fp_df.iloc[1:]
    if fp_df.study_id.iloc[0] in [3156269631] and fp_df.instance_number.iloc[0] == 12:
        # same
        fp_df = fp_df.iloc[:4]
    if fp_df.study_id.iloc[0] in [364930790] and fp_df.instance_number.iloc[0] == 13:
        # same
        fp_df = fp_df.iloc[1:]
    if fp_df.study_id.iloc[0] in [364930790] and fp_df.instance_number.iloc[0] == 14:
        # same
        fp_df = fp_df.iloc[:3]
    if 2 in fp_df.level.value_counts().tolist():
        break
    tmp_ann = {
        "filepath": fp,
        "coords": fp_df[["x", "y"]].values, 
        "labels": fp_df["label"].values,
        "fold": folds_dict[fp_df.study_id.iloc[0]]
    }
    annotations.append(tmp_ann)

for row in negative_foramen_df.itertuples():
    tmp_ann = {
        "filepath": row.filepath,
        "coords": np.zeros((0, 2)),
        "labels": [],
        "fold": folds_dict[row.study_id]
    }
    annotations.append(tmp_ann)

with open("../../data/train_foramen_seg_annotations.pkl", "wb") as f:
    pickle.dump(annotations, f)
