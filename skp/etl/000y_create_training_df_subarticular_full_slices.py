import glob
import os
import pandas as pd


folds_df = pd.read_csv("../../data/folds_cv5.csv")
train_df = pd.read_csv("../../data/train_narrow.csv")

subarticular = train_df.loc[train_df.condition == "subarticular"]
rt = subarticular.loc[subarticular.laterality == "R"]
lt = subarticular.loc[subarticular.laterality == "L"]

rt_cols = list(rt.columns)
rt_cols = [f"rt_{c}" if c not in ["study_id", "level"] else c for c in rt_cols]
lt_cols = list(lt.columns)
lt_cols = [f"lt_{c}" if c not in ["study_id", "level"] else c for c in lt_cols]

rt.columns = rt_cols
lt.columns = lt_cols

df = rt.merge(lt, on=["study_id", "level"])
df = df.merge(folds_df, on="study_id")
df["filepath"] = df.study_id.astype("str") + "_" + df.level + ".png"
df.to_csv("../../data/train_subarticular_full_slice.csv", index=False)
