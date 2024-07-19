import os
import pandas as pd 


features = os.listdir("../../data/train_foramen_dist_coord_features/fold0/")
features = [_ for _ in features if "features" in _]
labels = [_.replace("features", "labels") for _ in features]

df = pd.DataFrame({"features": features, "labels": labels})
df["study_id"] = df.features.apply(lambda x: int(x.split("_")[0]))
folds_df = pd.read_csv("../../data/folds_cv5.csv")
df = df.merge(folds_df, on="study_id")

df.to_csv("../../data/train_foramen_dist_coord_features.csv", index=False)
