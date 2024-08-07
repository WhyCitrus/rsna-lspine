import numpy as np
import pandas as pd


desc_df = pd.read_csv("../../data/train_series_descriptions.csv")
ax_t2_series = desc_df.loc[desc_df.series_description == "Axial T2"]
coords_df = pd.read_csv("../../data/train_label_coordinates.csv")
coords_df = coords_df.loc[coords_df.condition.apply(lambda x: "Subarticular" in x)]
coords_df["study_level_condition"] = coords_df.study_id.astype("str") + "-" + coords_df.condition.apply(lambda x: x[0]) + "-" + coords_df.level.apply(lambda x: x.replace("/", "_")) + "-subarticular"

# How many studies have more than 1 axial T2 series?
ax_t2_counts = pd.DataFrame(ax_t2_series.study_id.value_counts()).reset_index()
len(ax_t2_counts.loc[ax_t2_counts["count"] > 1])
# 335

##############################
# STUDIES WITH ONLY 1 SERIES #
##############################

# Of studies with 1 axial T2 series, how many are missing levels? (according to coordinates)
one_series = ax_t2_counts.loc[ax_t2_counts["count"] == 1].study_id.tolist()
one_series_coords = coords_df.loc[coords_df.study_id.isin(one_series)]

missing_levels = []
for study_id, study_df in one_series_coords.groupby("study_id"):
	assert len(study_df.series_id.unique()) == 1
	if len(np.unique(study_df.condition + study_df.level)) < 10:
		missing_levels.append(study_id)

len(missing_levels)
# 158

# Which ones are actually missing levels on both sides versus just one side?
missing_levels_both_sides = []
for study_id, study_df in one_series_coords.groupby("study_id"):
	if study_id not in missing_levels:
		continue
	if len(study_df.level.unique()) < 5:
		missing_levels_both_sides.append(study_id)

len(missing_levels_both_sides)
# 152 

# Which levels are missing?
missing_which_levels = []
levels = one_series_coords.level.unique().tolist()
for study_id, study_df in one_series_coords.groupby("study_id"):
	if study_id not in missing_levels_both_sides:
		continue
	levels_present = study_df.level.unique().tolist()
	missing_which_levels.extend(list(set(levels) - set(levels_present)))

np.unique(missing_which_levels, return_counts=True)
# (array(['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1'], dtype='<U5'), array([144,  79,   3,   3,   7]))

# Does this correlate with what is missing in the stenosis grading labels?
train_df = pd.read_csv("../../data/train_narrow.csv")
train_df = train_df.loc[train_df.condition == "subarticular"]
train_df["study_level_condition"] = train_df.study_id.astype("str") + "-" + train_df.laterality + "-" + train_df.level + "-" + train_df.condition

discordant = []
for study_id, study_df in one_series_coords.groupby("study_id"):
	if study_id not in missing_levels_both_sides:
		continue
	tmp_train_df = train_df.loc[train_df.study_id == study_id]
	if len(list(set(tmp_train_df.study_level_condition) - set(study_df.study_level_condition))) > 0:
		discordant.append(study_id)

# Only one discordant study: 3303545110
# Coords only available for L5/S1
# However, stenosis grading available for all levels (all normal_mild)
# So in general, if missing coordinates, stenosis grading also missing

# In summary,
# 1,640/1,975 studies have only 1 axial T2 series
# Of these, 152 are missing levels
# Most of which are L1/L2 and L2/L2

##########################
# STUDIES WITH >1 SERIES #
##########################

multiple_series = ax_t2_counts.loc[ax_t2_counts["count"] > 1].study_id.tolist()
multiple_series_coords = coords_df.loc[coords_df.study_id.isin(multiple_series)]

# How many of these studies only have 1 series labeled with coordinates?
one_series_labeled = []
for study_id, study_df in multiple_series_coords.groupby("study_id"):
	if len(study_df.series_id.unique()) == 1:
		one_series_labeled.append(study_id)

len(one_series_labeled)
# 0

# How many of these studies are missing levels across all series in the study?
missing_levels = []
for study_id, study_df in multiple_series_coords.groupby("study_id"):
	assert len(study_df.series_id.unique()) > 1
	if len(np.unique(study_df.condition + study_df.level)) < 10:
		missing_levels.append(study_id)

len(missing_levels)
# 18

# Which ones are actually missing levels on both sides versus just one side?
missing_levels_both_sides = []
for study_id, study_df in multiple_series_coords.groupby("study_id"):
	if study_id not in missing_levels:
		continue
	if len(study_df.level.unique()) < 5:
		missing_levels_both_sides.append(study_id)

len(missing_levels_both_sides)
# 17

# Which levels are missing?
missing_which_levels = []
levels = one_series_coords.level.unique().tolist()
for study_id, study_df in multiple_series_coords.groupby("study_id"):
	if study_id not in missing_levels_both_sides:
		continue
	levels_present = study_df.level.unique().tolist()
	missing_which_levels.extend(list(set(levels) - set(levels_present)))

np.unique(missing_which_levels, return_counts=True)
# (array(['L1/L2', 'L2/L3'], dtype='<U5'), array([17,  3]))

# How many of these studies have 1 series labeled with right side and one series labeled with left side? 
multiple_series_coords["laterality"] = multiple_series_coords.condition.apply(lambda x: x.split()[0])
each_side_different_series = []
for study_id, study_df in multiple_series_coords.groupby("study_id"):
	num_sides_present_in_each_series = []
	for series_id, series_df in study_df.groupby("series_id"):
		num_sides_present_in_each_series.append(len(series_df.laterality.unique()))
	if np.sum(num_sides_present_in_each_series) == len(study_df.series_id.unique()):
		each_side_different_series.append(study_id)

len(each_side_different_series)
# 141

# How many of these studies have more than 2 series in coordinates labels?
more_than_two_series = []
for study_id, study_df in multiple_series_coords.groupby("study_id"):
	if len(study_df.series_id.unique()) > 2:
		more_than_two_series.append(study_id)

len(more_than_two_series)
# 29


