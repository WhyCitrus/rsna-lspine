"""
Each axial slice has both left and right subarticular boxes.
Only includes levels where the ground truth left and right subarticular coordinate instance numbers
differ by 0 or 1. 
"""
import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm


df = pd.read_csv("../../data/train_predicting_axial_subarticular_coords_v2.csv")
df["w"] = 0.1 * df.cols
df["h"] = 0.1 * df.rows

folds_df = pd.read_csv("../../data/folds_cv5.csv")
folds_dict = {row.study_id: row.fold for row in folds_df.itertuples()}

annotations = []
for row in df.itertuples():
    # x1, y1, x2, y2
    lt_box = [row.lt_x - row.w / 2, row.lt_y - row.h / 2, row.lt_x + row.w / 2, row.lt_y + row.h / 2]
    rt_box = [row.rt_x - row.w / 2, row.rt_y - row.h / 2, row.rt_x + row.w / 2, row.rt_y + row.h / 2]
    tmp_ann = {
        "filepath": row.filepath,
        "bboxes": np.stack([lt_box, rt_box]).astype("int"),
        "labels": np.asarray([0, 1]),
        "fold": folds_dict[row.study_id]
    }
    annotations.append(tmp_ann)

with open("../../data/train_subarticular_bboxes_v2.pkl", "wb") as f:
    pickle.dump(annotations, f)
