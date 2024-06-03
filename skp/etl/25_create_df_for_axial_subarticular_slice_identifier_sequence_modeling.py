import pandas as pd

from collections import defaultdict


df = pd.read_csv("../../data/train_label_coordinates.csv")
meta_df = pd.read_csv("../../data/dicom_metadata.csv")
subart_df = df.loc[df.condition.apply(lambda x: "Subarticular" in x)]

# Only take fully labeled series
df_list = []
for series_id, series_df in subart_df.groupby("series_id"):
	if len(series_df) == 10:
		df_list.append(series_df)

subart_df = pd.concat(df_list)
study_series_instance_to_level = {f"{row.study_id}-{row.series_id}-{row.instance_number}": row.level.replace("/", "_").lower() for row in subart_df.itertuples()}

new_df_list = []
for series_id, series_df in subart_df.groupby("series_id"):
	tmp_df = meta_df.loc[meta_df.series_id == series_id]
	new_df_dict = defaultdict(list)
	for row in tmp_df.itertuples():
		study_series_instance = f"{row.study_id}-{row.series_id}-{row.instance_number}"
		new_df_dict["study_id"].append(row.study_id)
		new_df_dict["series_id"].append(row.series_id)
		new_df_dict["instance_number"].append(row.instance_number)
		if study_series_instance not in [*study_series_instance_to_level]:
			for lvl in ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]:
				new_df_dict[lvl].append(0)
		else:
			for lvl in ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]:
				if lvl == study_series_instance_to_level[study_series_instance]:
					new_df_dict[lvl].append(1)
				else:
					new_df_dict[lvl].append(0)
	new_df_list.append(pd.DataFrame(new_df_dict))

new_df = pd.concat(new_df_list)
new_df = new_df.sort_values(["study_id", "series_id", "instance_number"], ascending=True).reset_index(drop=True)

folds_df = pd.read_csv("../../data/folds_cv5.csv")
new_df = new_df.merge(folds_df, on="study_id")
new_df.to_csv("../../data/train_axial_subarticular_slice_identifier_sequence.csv", index=False)


