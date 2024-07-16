import glob
import numpy as np
import os
import pandas as pd

from collections import defaultdict


stack_slices = glob.glob("../../data/train_cropped_foramina_blocks/*/*")
stack_slice_dict = defaultdict(list)
for f in stack_slices:
	stack_slice_dict[os.path.dirname(f)].append(f)

num_slices_per_stack = [len(v) for v in stack_slice_dict.values()]

np.percentile(num_slices_per_stack, [0, 10, 25, 50, 75, 90, 100])
# array([11., 14., 15., 17., 19., 21., 29.])

train_df = pd.read_csv("../../data/train_narrow.csv")
subart_df = train_df.loc[train_df.condition == "foraminal"]

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

bilat_df.to_csv("../../data/train_foramina_blocks_kfold.csv", index=False)
