import numpy as np
import pandas as pd


coords_df = pd.read_csv("../../data/train_label_coordinates.csv")
spinal_df = coords_df.loc[coords_df.condition.apply(lambda x: "Spinal" in x)]
meta_df = pd.read_csv("../../data/dicom_metadata.csv")
meta_df = meta_df.loc[meta_df.series_id.isin(spinal_df.series_id.tolist())]

instance_to_position_index_dict = {}
for series_id, series_df in meta_df.groupby("series_id"):
    series_df = series_df.sort_values("ImagePositionPatient0", ascending=True)
    series_df["position_index"] = np.arange(len(series_df))
    instance_to_position_index_dict.update({f"{series_id}_{row.instance_number}": row.position_index for row in series_df.itertuples()})

num_slices_per_series = {series_id: len(series_df) for series_id, series_df in meta_df.groupby("series_id")}

df_list = []
incomplete_series = []
for series_id, series_df in spinal_df.groupby("series_id"):
	if len(series_df.level.unique()) != 5:
		incomplete_series.append(series_id)
		continue
	df_list.append(series_df)

spinal_df = pd.concat(df_list)
spinal_df["series_instance"] = spinal_df.series_id.astype("str") + "_" + spinal_df.instance_number.astype("str")
spinal_df["position_index"] = spinal_df.series_instance.map(instance_to_position_index_dict)
# fix mislabeled instance
spinal_df.loc[33244, "position_index"] = 10
spinal_df["num_slices"] = spinal_df.series_id.map(num_slices_per_series)

num_slices = 12
spinal_df["slice_start"] = (spinal_df.num_slices // 2 - num_slices // 2) 
spinal_df["slice_end"] = spinal_df.slice_start + num_slices
spinal_df["slice_start"] = spinal_df.slice_start.apply(lambda x: max(0, x))
spinal_df["position_index_shifted"] = spinal_df.position_index - spinal_df.slice_start
spinal_df["slice_rescaled"] = spinal_df.position_index_shifted / num_slices
spinal_df = spinal_df.merge(meta_df, on=["study_id", "series_id", "instance_number"])
spinal_df["x_rescaled"] = spinal_df.x / spinal_df.cols
spinal_df["y_rescaled"] = spinal_df.y / spinal_df.rows

df_list = []
for series_id, series_df in spinal_df.groupby("series_id"):
	series_df["level"] = series_df.level.apply(lambda x: x.replace("/", "_").lower())
	tmp_dict = {}
	tmp_dict["study_id"] = series_df.study_id.values[0]
	tmp_dict["series_id"] = series_df.series_id.values[0]
	tmp_dict["slice_start"] = series_df.slice_start.values[0]
	tmp_dict["slice_end"] = series_df.slice_end.values[0]
	tmp_dict["num_slices"] = series_df.num_slices.values[0]
	for row_idx, row in series_df.iterrows():
		tmp_dict[f"{row.level}_instance_number"] = row.instance_number
		tmp_dict[f"{row.level}_position_index"] = row.position_index
		tmp_dict[f"{row.level}_position_index_shifted"] = row.position_index_shifted
		tmp_dict[f"{row.level}_slice_rescaled"] = row.slice_rescaled
		tmp_dict[f"{row.level}_x_rescaled"] = row.x_rescaled
		tmp_dict[f"{row.level}_y_rescaled"] = row.y_rescaled
	df_list.append(pd.DataFrame(tmp_dict, index=[0]))

df = pd.concat(df_list)
df["series_folder"] = df.study_id.astype("str") + "/" + df.series_id.astype("str")
folds_df = pd.read_csv("../../data/folds_cv5.csv")
df = df.merge(folds_df, on="study_id")

train_df = pd.read_csv("../../data/train_narrow.csv")
train_df = train_df.loc[train_df.condition == "spinal"]
df_list = []
for study_id, study_df in train_df.groupby("study_id"):
	if study_id not in df.study_id.tolist():
		continue
	study_df = study_df.sort_values("level")
	tmp_dict = {}
	tmp_dict["study_id"] = study_id
	for row_idx, row in study_df.iterrows():
		tmp_dict[f"{row.level.lower()}_normal_mild"] = row.normal_mild
		tmp_dict[f"{row.level.lower()}_moderate"] = row.moderate
		tmp_dict[f"{row.level.lower()}_severe"] = row.severe
	df_list.append(pd.DataFrame(tmp_dict, index=[0]))

train_df_wide = pd.concat(df_list)
df = df.merge(train_df_wide, on="study_id")
df.to_csv("../../data/train_spinal_3d_whole_series.csv", index=False)
