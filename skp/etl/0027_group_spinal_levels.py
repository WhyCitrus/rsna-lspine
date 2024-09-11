import pandas as pd
import pickle


df = pd.read_csv("../../data/train_gt_spinal_with_augs_kfold.csv")

levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
levels = [l.upper() for l in levels]
annotations = []
for study_id, study_df in df.groupby("study_id"):
    if len(study_df.level.unique()) != 5:
        continue
    study_df = study_df.sort_values("level")
    study_dict = {level: study_df.loc[study_df.level == level, "filepath"].tolist() for level in levels}
    label_arr = np.stack([study_df.loc[study_df.level == level, ["normal_mild", "moderate", "severe"]].iloc[0].values for level in levels])
    study_dict = {"files": study_dict}
    study_dict["labels"] = label_arr
    study_dict["fold"] = study_df.fold.values[0]
    annotations.append(study_dict)

with open("../../data/train_grouped_spinal_annotations.pkl", "wb") as f:
    pickle.dump(annotations, f)
