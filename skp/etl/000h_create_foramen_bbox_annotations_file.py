import pandas as pd
import pickle

from tqdm import tqdm


df = pd.read_csv("../../data/train_label_coordinates.csv")
meta_df = pd.read_csv("../../data/dicom_metadata.csv")
df = df.merge(meta_df, on=["study_id", "series_id", "instance_number"])
df = df.loc[df.condition.apply(lambda x: "Foraminal" in x)]
df["w"] = 0.1 * df.cols
df["h"] = 0.1 * df.rows
df["x1"] = (df.x - df.w / 2).astype("int")
df["x2"] = (df.x + df.w / 2).astype("int")
df["y1"] = (df.y - df.h / 2).astype("int")
df["y2"] = (df.y + df.h / 2).astype("int")
df["label"] = pd.Categorical(df.level).codes
df["filepath"] = df.study_id.astype("str") + "/" + df.series_id.astype("str") + "/" + df.instance_number.apply(lambda x: f"IM{x:06d}.png")

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
        "bboxes": fp_df[["x1", "y1", "x2", "y2"]].values, 
        "labels": fp_df["label"].values,
        "fold": folds_dict[fp_df.study_id.iloc[0]]
    }
    annotations.append(tmp_ann)

with open("../../data/train_foramen_bboxes.pkl", "wb") as f:
    pickle.dump(annotations, f)


df = pd.read_csv("../../data/train_label_coordinates.csv")
meta_df = pd.read_csv("../../data/dicom_metadata.csv")
df = df.merge(meta_df, on=["study_id", "series_id", "instance_number"])
df = df.loc[df.condition.apply(lambda x: "Foraminal" in x)]
df["w"] = 0.05 * df.cols
df["h"] = 0.05 * df.rows
df["x1"] = (df.x - df.w / 2).astype("int")
df["x2"] = (df.x + df.w / 2).astype("int")
df["y1"] = (df.y - df.h / 2).astype("int")
df["y2"] = (df.y + df.h / 2).astype("int")
df["label"] = pd.Categorical(df.level).codes
df["filepath"] = df.study_id.astype("str") + "/" + df.series_id.astype("str") + "/" + df.instance_number.apply(lambda x: f"IM{x:06d}.png")

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
        "bboxes": fp_df[["x1", "y1", "x2", "y2"]].values, 
        "labels": fp_df["label"].values,
        "fold": folds_dict[fp_df.study_id.iloc[0]]
    }
    annotations.append(tmp_ann)

with open("../../data/train_foramen_bboxes_smaller.pkl", "wb") as f:
    pickle.dump(annotations, f)
