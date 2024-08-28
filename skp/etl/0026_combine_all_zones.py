import pandas as pd


foramen_df = pd.read_csv("../../data/train_gt_foraminal_with_augs_kfold.csv")
spinal_df = pd.read_csv("../../data/train_gt_spinal_with_augs_kfold.csv")
subarticular_df = pd.read_csv("../../data/train_gt_subarticular_with_augs_kfold.csv")
ax_spinal_df = pd.read_csv("../../data/train_gt_axial_spinal_with_augs_kfold.csv")

foramen_df["filepath"] = "train_crops_gt_with_augs/foraminal/" + foramen_df.filepath
spinal_df["filepath"] = "train_crops_gt_with_augs/spinal/" + spinal_df.filepath
subarticular_df["filepath"] = "train_crops_gt_with_augs/subarticular/" + subarticular_df.filepath
ax_spinal_df["filepath"] = "train_axial_spinal_crops_gt_with_augs/" + ax_spinal_df.filepath

foramen_df["normal_mild_foramen"] = foramen_df.normal_mild
foramen_df["moderate_foramen"] = foramen_df.moderate
foramen_df["severe_foramen"] = foramen_df.severe
foramen_df["normal_mild_spinal"] = 0
foramen_df["moderate_spinal"] = 0
foramen_df["severe_spinal"] = 0
foramen_df["normal_mild_subarticular"] = 0
foramen_df["moderate_subarticular"] = 0
foramen_df["severe_subarticular"] = 0

spinal_df["normal_mild_foramen"] = 0
spinal_df["moderate_foramen"] = 0
spinal_df["severe_foramen"] = 0
spinal_df["normal_mild_spinal"] = spinal_df.normal_mild
spinal_df["moderate_spinal"] = spinal_df.moderate
spinal_df["severe_spinal"] = spinal_df.severe
spinal_df["normal_mild_subarticular"] = 0
spinal_df["moderate_subarticular"] = 0
spinal_df["severe_subarticular"] = 0

subarticular_df["normal_mild_foramen"] = 0
subarticular_df["moderate_foramen"] = 0
subarticular_df["severe_foramen"] = 0
subarticular_df["normal_mild_spinal"] = 0
subarticular_df["moderate_spinal"] = 0
subarticular_df["severe_spinal"] = 0
subarticular_df["normal_mild_subarticular"] = subarticular_df.normal_mild
subarticular_df["moderate_subarticular"] = subarticular_df.moderate
subarticular_df["severe_subarticular"] = subarticular_df.severe

ax_spinal_df["normal_mild_foramen"] = 0
ax_spinal_df["moderate_foramen"] = 0
ax_spinal_df["severe_foramen"] = 0
ax_spinal_df["normal_mild_spinal"] = ax_spinal_df.normal_mild
ax_spinal_df["moderate_spinal"] = ax_spinal_df.moderate
ax_spinal_df["severe_spinal"] = ax_spinal_df.severe
ax_spinal_df["normal_mild_subarticular"] = 0
ax_spinal_df["moderate_subarticular"] = 0
ax_spinal_df["severe_subarticular"] = 0

foramen_df["sampling_weight"] = 1 / len(foramen_df)
spinal_df["sampling_weight"] = 1 / len(spinal_df)
subarticular_df["sampling_weight"] = 1 / len(subarticular_df)
ax_spinal_df["sampling_weight"] = 1 / len(ax_spinal_df)

foramen_df["unique_identifier"] = "FORAMEN_" + foramen_df.unique_identifier
spinal_df["unique_identifier"] = "SPINAL_" + spinal_df.unique_identifier
subarticular_df["unique_identifier"] = "SUBART_" + subarticular_df.unique_identifier
ax_spinal_df["unique_identifier"] = "SPINAL_" + ax_spinal_df.unique_identifier

df = pd.concat([foramen_df, spinal_df, subarticular_df, ax_spinal_df])
df["unique_id"] = pd.Categorical(df.unique_identifier).codes
df.loc[df.unique_identifier.apply(lambda x: "SPINAL" in x), "unique_id"] += 100000
df.loc[df.unique_identifier.apply(lambda x: "SUBART" in x), "unique_id"] += 200000

df.to_csv("../../data/train_gt_all_with_augs_sampling_weight_kfold.csv", index=False)
