import numpy as np
import pandas as pd

from collections import defaultdict


meta_df = pd.read_csv("../../data/dicom_metadata.csv")
desc_df = pd.read_csv("../../data/train_series_descriptions.csv")
coords_df = pd.read_csv("../../data/train_label_coordinates.csv")
coords_df = coords_df.loc[~coords_df.condition.apply(lambda x: "Foraminal" in x)]

df_list = []
for study_id, study_df in coords_df.groupby("study_id"):
	subart_series = study_df.loc[study_df.condition.apply(lambda x: "Subarticular" in x)]
	rt_subart = subart_series.loc[subart_series.condition.apply(lambda x: "Right" in x)]
	lt_subart = subart_series.loc[subart_series.condition.apply(lambda x: "Left" in x)]
	spinal = study_df.loc[study_df.condition.apply(lambda x: "Spinal" in x)]
	subart = rt_subart.merge(lt_subart, on=["study_id", "level"], suffixes=["_rt", "_lt"])
	if len(subart) != 5:
		continue
	# Assume the center of the spinal canal is between the two labeled subarticular zones
	xc, yc = (subart.x_rt + subart.x_lt) / 2, (subart.y_rt, subart.y_lt)
	# Then, calculate the of the x-coordinate distance from each zone to the center]
	# Which should be the same for both sides
	dist = xc - subart.x_rt
	# Now convert to distance in mm using PixelSpacing
	tmp_meta_df = meta_df.loc[meta_df.series_id == subart.series_id_rt.values[0]]
	# Sometimes there are more than 1 axial T2 series
	# We are assuming the pixel spacing stays the same
	pixel_spacing = tmp_meta_df.PixelSpacing0.values[0]
	dist = dist * pixel_spacing
	tmp_meta_df = meta_df.loc[meta_df.series_id == spinal.series_id.values[0]].sort_values("ImagePositionPatient0")
	spinal = spinal.sort_values("level").reset_index(drop=True)
	subart_dict = defaultdict(list)
	for row_idx, row in spinal.iterrows():
		tmp_slice = tmp_meta_df.loc[tmp_meta_df.instance_number == row.instance_number]
		tmp_dist = dist[row_idx]
		tmp_slice_position = tmp_slice.ImagePositionPatient0.iloc[0]
		rt_subart_position = tmp_slice_position - tmp_dist
		lt_subart_position = tmp_slice_position + tmp_dist
		rt_subart_slice = np.argmin(np.abs(tmp_meta_df.ImagePositionPatient0.values - rt_subart_position))
		lt_subart_slice = np.argmin(np.abs(tmp_meta_df.ImagePositionPatient0.values - lt_subart_position))
		subart_dict["instance_number_rt"].append(tmp_meta_df.iloc[rt_subart_slice].instance_number)
		subart_dict["instance_number_lt"].append(tmp_meta_df.iloc[lt_subart_slice].instance_number)
		subart_dict["level"].append(row.level)
		subart_dict["study_id"].append(study_id)
		subart_dict["series_id"].append(spinal.series_id.values[0])
		subart_dict["x"].append(row.x)
		subart_dict["y"].append(row.y)
	tmp_subart_df = pd.DataFrame(subart_dict)
	df_list.append(tmp_subart_df)

subart_df = pd.concat(df_list)
rt = subart_df.copy()
rt["laterality"] = "R"
rt["instance_number"] = rt.instance_number_rt
lt = subart_df.copy()
lt["laterality"] = "L"
lt["instance_number"] = lt.instance_number_lt
subart_df = pd.concat([rt, lt])
subart_df.to_csv("../../data/sag_t2_subarticular_coords_based_on_spinal_and_mean_dist.csv", index=False)
