import pandas as pd


# FORAMEN
df = pd.read_csv("../../data/train_foramen_distance_each_level_with_foramen_coords_and_ignore.csv")

exclude = [
	3742728457, 757619082, 364930790, 2410494888, 3156269631, 1085426528, 1647904243, 2135829458, 2626030939, 
	1879696087, 3495818564, 1395773918, 2388577668, 2530679352, 2662989538,
]

error_df = pd.read_csv("../../notebooks/large_foramina_slice_instance_error_dist_coord_seg_v3.csv")
error_df = error_df.loc[~error_df.study_id.isin(exclude)]

difficult_studies = df.loc[df.study_id.isin(error_df.study_id.tolist())]
# upsample
df = pd.concat([df] + [difficult_studies] * 3)

df.to_csv("../../data/train_foramen_dist_each_level_with_coords_and_ignore_upsample_hard_cases.csv", index=False)


# SPINAL
df = pd.read_csv("../../data/train_spinal_distance_each_level_with_spinal_coords_and_ignore.csv")

exclude = [2937953357, 1973833645]

error_df = pd.read_csv("../../notebooks/large_spinal_slice_instance_error_dist_coord_seg_v3.csv")
error_df = error_df.loc[~error_df.study_id.isin(exclude)]

difficult_studies = df.loc[df.study_id.isin(error_df.study_id.tolist())]
# upsample
df = pd.concat([df] + [difficult_studies] * 9)

df.to_csv("../../data/train_spinal_dist_each_level_with_coords_and_ignore_upsample_hard_cases.csv", index=False)

