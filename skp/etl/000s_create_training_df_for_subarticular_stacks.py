import glob
import numpy as np
import os
import pandas as pd

from collections import defaultdict


stack_slices = glob.glob("../../data/train_subarticular_stacks/*/*")
stack_slice_dict = defaultdict(list)
for f in stack_slices:
	stack_slice_dict[os.path.dirname(f)].append(f)

num_slices_per_stack = [len(v) for v in stack_slice_dict.values()]

np.percentile(num_slices_per_stack, [0, 10, 25, 50, 75, 90, 100])
# array([ 1.,  5.,  6.,  7.,  9., 11., 40.])

train_df = pd.read_csv("../../data/train_narrow.csv")
subart_df = train_df.loc[train_df.condition == "subarticular"]

rt_df = subart_df.loc[subart_df.laterality == "R"]
lt_df = subart_df.loc[subart_df.laterality == "L"]

rt_cols = list(rt_df.columns)
rt_cols = [f"rt_{c}" if c not in ["study_id", "level"] else c for c in rt_cols]
lt_cols = list(lt_df.columns)
lt_cols = [f"lt_{c}" if c not in ["study_id", "level"] else c for c in lt_cols]

rt_df.columns = rt_cols
lt_df.columns = lt_cols

bilat_df = rt_df.merge(lt_df, on=["study_id", "level"])
folds_df = pd.read_csv("../../data/folds_cv5.csv")

bilat_df = bilat_df.merge(folds_df, on="study_id")
bilat_df["series_folder"] = bilat_df.study_id.astype("str") + "_" + bilat_df.level

bilat_df.to_csv("../../data/train_subarticular_stacks_kfold.csv", index=False)

bilat_df["filepath"] = bilat_df.series_folder + ".png"

bilat_df.to_csv("../../data/train_subarticular_slices_v2_kfold.csv", index=False)
