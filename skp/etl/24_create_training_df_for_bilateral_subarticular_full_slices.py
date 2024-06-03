import pandas as pd


df = pd.read_csv("../../data/train_subarticular_crops_v4.csv")
relevant_cols = ["study_id", "grade", "level", "normal_mild", "moderate", "severe"]
left, right = df.loc[df.laterality == "L", relevant_cols], df.loc[df.laterality == "R", relevant_cols]
left.columns = [f"lt_{c}" if c not in ["study_id", "level"] else c for c in left.columns]
right.columns = [f"rt_{c}" if c not in ["study_id", "level"] else c for c in right.columns]

new_df = left.merge(right, on=["study_id", "level"])
folds_df = pd.read_csv("../../data/folds_cv5.csv")
new_df = new_df.merge(folds_df, on="study_id")
new_df["filepath"] = df.study_id.astype("str") + "_" + df.level + ".png"
new_df.to_csv("../../data/train_subarticular_bilateral_full_slices.csv", index=False)
