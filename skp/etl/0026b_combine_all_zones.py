import numpy as np
import pandas as pd


foramen_df = pd.read_csv("../../data/train_gt_foraminal_with_augs_kfold.csv")
spinal_df = pd.read_csv("../../data/train_gt_spinal_with_augs_kfold.csv")
subarticular_df = pd.read_csv("../../data/train_gt_subarticular_with_augs_kfold.csv")
ax_spinal_df = pd.read_csv("../../data/train_gt_axial_spinal_with_augs_kfold.csv")

foramen_df["filepath"] = "train_crops_gt_with_augs/foraminal/" + foramen_df.filepath
spinal_df["filepath"] = "train_crops_gt_with_augs/spinal/" + spinal_df.filepath
subarticular_df["filepath"] = "train_crops_gt_with_augs/subarticular/" + subarticular_df.filepath
ax_spinal_df["filepath"] = "train_axial_spinal_crops_gt_with_augs/" + ax_spinal_df.filepath

lt_subart = subarticular_df.loc[subarticular_df.laterality == "L"][["study_id", "level", "normal_mild", "moderate", "severe"]].drop_duplicates()
rt_subart = subarticular_df.loc[subarticular_df.laterality == "R"][["study_id", "level", "normal_mild", "moderate", "severe"]].drop_duplicates()
bilat_subart = lt_subart.merge(rt_subart, on=["study_id", "level"], suffixes=["_lt", "_rt"])
bilat_subart["grade_rt"] = np.argmax(bilat_subart[["normal_mild_rt", "moderate_rt", "severe_rt"]].values, axis=1)
bilat_subart["grade_lt"] = np.argmax(bilat_subart[["normal_mild_lt", "moderate_lt", "severe_lt"]].values, axis=1)
bilat_subart["grade"] = np.max((bilat_subart[["grade_rt", "grade_lt"]].values), axis=1)
bilat_subart["normal_mild"] = 0
bilat_subart["moderate"] = 0
bilat_subart["severe"] = 0
bilat_subart.loc[bilat_subart.grade == 0, "normal_mild"] = 1
bilat_subart.loc[bilat_subart.grade == 1, "moderate"] = 1
bilat_subart.loc[bilat_subart.grade == 2, "severe"] = 1
bilat_subart["unique_identifier"] = bilat_subart.study_id.astype("str") + "_" + bilat_subart.level

spinal_dict = {
	"normal_mild": 
		{
			row.unique_identifier: row.normal_mild for row in spinal_df.itertuples()
		},
	"moderate":
		{
			row.unique_identifier: row.moderate for row in spinal_df.itertuples()
		},
	"severe":
		{
			row.unique_identifier: row.severe for row in spinal_df.itertuples()
		}
}

subart_dict = {
	"normal_mild": 
		{
			row.unique_identifier: row.normal_mild for row in bilat_subart.itertuples()
		},
	"moderate":
		{
			row.unique_identifier: row.moderate for row in bilat_subart.itertuples()
		},
	"severe":
		{
			row.unique_identifier: row.severe for row in bilat_subart.itertuples()
		}
}

subarticular_df["unique_identifier2"] = subarticular_df.study_id.astype("str") + "_" + subarticular_df.level

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
subarticular_df["normal_mild_spinal"] = subarticular_df.unique_identifier2.map(spinal_dict["normal_mild"])
subarticular_df["moderate_spinal"] = subarticular_df.unique_identifier2.map(spinal_dict["moderate"])
subarticular_df["severe_spinal"] = subarticular_df.unique_identifier2.map(spinal_dict["severe"])
subarticular_df["normal_mild_subarticular"] = subarticular_df.normal_mild
subarticular_df["moderate_subarticular"] = subarticular_df.moderate
subarticular_df["severe_subarticular"] = subarticular_df.severe
subarticular_df["is_missing"] = subarticular_df.normal_mild_spinal.isna().astype("int") + subarticular_df.moderate_spinal.isna().astype("int") + subarticular_df.severe_spinal.isna().astype("int")
subarticular_df = subarticular_df.loc[subarticular_df.is_missing == 0]
del subarticular_df["is_missing"]
del subarticular_df["unique_identifier2"]

ax_spinal_df["normal_mild_foramen"] = 0
ax_spinal_df["moderate_foramen"] = 0
ax_spinal_df["severe_foramen"] = 0
ax_spinal_df["normal_mild_spinal"] = ax_spinal_df.normal_mild
ax_spinal_df["moderate_spinal"] = ax_spinal_df.moderate
ax_spinal_df["severe_spinal"] = ax_spinal_df.severe
ax_spinal_df["normal_mild_subarticular"] = ax_spinal_df.unique_identifier.map(subart_dict["normal_mild"])
ax_spinal_df["moderate_subarticular"] = ax_spinal_df.unique_identifier.map(subart_dict["moderate"])
ax_spinal_df["severe_subarticular"] = ax_spinal_df.unique_identifier.map(subart_dict["severe"])
ax_spinal_df["is_missing"] = ax_spinal_df.normal_mild_subarticular.isna().astype("int") + ax_spinal_df.moderate_subarticular.isna().astype("int") + ax_spinal_df.severe_subarticular.isna().astype("int")
ax_spinal_df = ax_spinal_df.loc[ax_spinal_df.is_missing == 0]
del ax_spinal_df["is_missing"]

foramen_df["unique_identifier"] = "FORAMEN_" + foramen_df.unique_identifier
spinal_df["unique_identifier"] = "SPINAL_" + spinal_df.unique_identifier
subarticular_df["unique_identifier"] = "SUBART_" + subarticular_df.unique_identifier
ax_spinal_df["unique_identifier"] = "SPINAL_" + ax_spinal_df.unique_identifier

df = pd.concat([foramen_df, spinal_df, subarticular_df, ax_spinal_df])
df["unique_id"] = pd.Categorical(df.unique_identifier).codes
df.loc[df.unique_identifier.apply(lambda x: "SPINAL" in x), "unique_id"] += 100000
df.loc[df.unique_identifier.apply(lambda x: "SUBART" in x), "unique_id"] += 200000

df.to_csv("../../data/train_gt_all_with_augs_kfold2.csv", index=False)
