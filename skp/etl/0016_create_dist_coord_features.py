import glob
import os
import pandas as pd


foramen_feats = glob.glob(os.path.join("../../data/train_foramen_dist_coord_features_v3/fold0/*features.npy"))
spinal_feats = glob.glob(os.path.join("../../data/train_spinal_dist_coord_features_v3/fold0/*features.npy"))
subarticular_feats = glob.glob(os.path.join("../../data/train_subarticular_dist_coord_features_v3/fold0/*features.npy"))

folds_df = pd.read_csv("../../data/folds_cv5.csv")

foramen_df = pd.DataFrame({"features": foramen_feats})
foramen_df["features"] = foramen_df.features.apply(lambda x: os.path.basename(x))
foramen_df["labels"] = foramen_df.features.apply(lambda x: x.replace("features", "labels"))
foramen_df["study_id"] = foramen_df.features.apply(lambda x: x.split("_")[0]).astype("int")
foramen_df["series_id"] = foramen_df.features.apply(lambda x: x.split("_")[1]).astype("int")
foramen_df = foramen_df.merge(folds_df, on="study_id")

spinal_df = pd.DataFrame({"features": spinal_feats})
spinal_df["features"] = spinal_df.features.apply(lambda x: os.path.basename(x))
spinal_df["labels"] = spinal_df.features.apply(lambda x: x.replace("features", "labels"))
spinal_df["study_id"] = spinal_df.features.apply(lambda x: x.split("_")[0]).astype("int")
spinal_df["series_id"] = spinal_df.features.apply(lambda x: x.split("_")[1]).astype("int")
spinal_df = spinal_df.merge(folds_df, on="study_id")

subarticular_df = pd.DataFrame({"features": subarticular_feats})
subarticular_df["features"] = subarticular_df.features.apply(lambda x: os.path.basename(x))
subarticular_df["labels"] = subarticular_df.features.apply(lambda x: x.replace("features", "labels"))
subarticular_df["study_id"] = subarticular_df.features.apply(lambda x: x.split("_")[0]).astype("int")
subarticular_df["series_id"] = subarticular_df.features.apply(lambda x: x.split("_")[1]).astype("int")
subarticular_df = subarticular_df.merge(folds_df, on="study_id")

foramen_df.to_csv("../../data/train_foramen_dist_coord_features_v3.csv", index=False)
spinal_df.to_csv("../../data/train_spinal_dist_coord_features_v3.csv", index=False)
subarticular_df.to_csv("../../data/train_subarticular_dist_coord_features_v3.csv", index=False)
