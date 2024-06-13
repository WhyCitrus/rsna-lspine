import pandas as pd
import pickle

from tqdm import tqdm


df = pd.read_csv("../../data/train_label_coordinates.csv")
meta_df = pd.read_csv("../../data/dicom_metadata.csv")
df = df.merge(meta_df, on=["study_id", "series_id", "instance_number"])
df = df.loc[df.condition.apply(lambda x: "Subarticular" in x)]
df["w"] = 0.075 * df.cols
df["h"] = 0.075 * df.rows
df["x1"] = (df.x - df.w / 2).astype("int")
df["x2"] = (df.x + df.w / 2).astype("int")
df["y1"] = (df.y - df.h / 2).astype("int")
df["y2"] = (df.y + df.h / 2).astype("int")
df["label"] = pd.Categorical(df.condition).codes
df["filepath"] = df.study_id.astype("str") + "/" + df.series_id.astype("str") + "/" + df.instance_number.apply(lambda x: f"IM{x:06d}.png")

folds_df = pd.read_csv("../../data/folds_cv5.csv")
folds_dict = {row.study_id: row.fold for row in folds_df.itertuples()}

annotations = []
for fp, fp_df in tqdm(df.groupby("filepath"), total=len(df.filepath.unique())):
    if 2 in fp_df.condition.value_counts().tolist():
        break
    tmp_ann = {
        "filepath": fp,
        "bboxes": fp_df[["x1", "y1", "x2", "y2"]].values, 
        "labels": fp_df["label"].values,
        "fold": folds_dict[fp_df.study_id.iloc[0]]
    }
    annotations.append(tmp_ann)

with open("../../data/train_subarticular_bboxes.pkl", "wb") as f:
    pickle.dump(annotations, f)
