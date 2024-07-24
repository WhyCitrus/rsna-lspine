import glob
import os
import pandas as pd


features = glob.glob("../../data/train_subarticular_dist_coord_features_v2/fold0/*features.npy")
features = [os.path.basename(_) for _ in features]
labels = [_.replace("features", "labels") for _ in features]

folds_df = pd.read_csv("../../data/folds_cv5.csv")
meta_df = pd.read_csv("../../data/dicom_metadata.csv")[["study_id", "series_id"]].drop_duplicates()
folds_df = folds_df.merge(meta_df, on="study_id")

df = pd.DataFrame({"features": features, "labels": labels})
df["series_id"] = df.features.apply(lambda x: x.split("_")[0]).astype("int")
df = df.merge(folds_df, on="series_id")

df.to_csv("../../data/train_subarticular_dist_coord_features.csv", index=False)
