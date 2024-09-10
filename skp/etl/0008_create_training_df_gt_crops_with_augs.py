import glob
import os
import pandas as pd


def create_training_df(image_dir, condition):
	images = glob.glob(os.path.join(image_dir, "*.png"))
	df = pd.DataFrame({"filepath": images})
	df["filepath"] = df.filepath.map(os.path.basename)
	df["study_id"] = df.filepath.apply(lambda x: x.split("_")[0]).astype("int")
	df["level"] = df.filepath.apply(lambda x: "_".join(x.split("_")[-4:-2]))
	df["laterality"] = df.filepath.apply(lambda x: x.split("_")[-5]) if "spinal" not in image_dir else None
	df = df.merge(folds_df, on="study_id")
	merge_cols = ["study_id", "level"]
	if "spinal" not in image_dir:
		merge_cols.append("laterality")
	else:
		del df["laterality"]
	df = df.merge(train_df.loc[train_df.condition == condition], on=merge_cols)
	if "spinal" not in image_dir:
		df["unique_identifier"] = df.study_id.astype("str") + "_" + df.laterality + "_" + df.level
	else:
		df["unique_identifier"] = df.study_id.astype("str") + "_" + df.level
	df["unique_id"] = pd.Categorical(df.unique_identifier).codes
	return df


folds_df = pd.read_csv("../../data/folds_cv5.csv")
train_df = pd.read_csv("../../data/train_narrow.csv")

gt_spinal = create_training_df("../../data/train_crops_gt_with_augs/spinal/", "spinal")
gt_axial_spinal = create_training_df("../../data/train_axial_spinal_crops_gt_with_augs/", "spinal")
gt_foraminal = create_training_df("../../data/train_crops_gt_with_augs/foraminal/", "foraminal")
gt_subarticular = create_training_df("../../data/train_crops_gt_with_augs/subarticular/", "subarticular")
gt_sag_subarticular = create_training_df("../../data/train_sag_t2_subarticular_crops_gt_with_augs_v2/", "subarticular")

gt_spinal.to_csv("../../data/train_gt_spinal_with_augs_kfold.csv", index=False)
gt_axial_spinal.to_csv("../../data/train_gt_axial_spinal_with_augs_kfold.csv", index=False)
gt_foraminal.to_csv("../../data/train_gt_foraminal_with_augs_kfold.csv", index=False)
gt_subarticular.to_csv("../../data/train_gt_subarticular_with_augs_kfold.csv", index=False)
gt_sag_subarticular.to_csv("../../data/train_gt_sag_subarticular_with_augs_kfold_v2.csv", index=False)


import glob
import os
import pandas as pd


folds_df = pd.read_csv("../../data/folds_cv5.csv")
train_df = pd.read_csv("../../data/train_narrow.csv")

images = glob.glob(os.path.join("../../data/train_sag_t2_subarticular_crops_gt/", "*"))
df = pd.DataFrame({"filepath": images})
df["filepath"] = df.filepath.map(os.path.basename)
df["study_id"] = df.filepath.apply(lambda x: x.split("_")[0]).astype("int")
df["level"] = df.filepath.apply(lambda x: "_".join(x.split("_")[-2:]))
df["laterality"] = df.filepath.apply(lambda x: x.split("_")[-3])
df = df.merge(folds_df, on="study_id")
df = df.merge(train_df.loc[train_df.condition == "subarticular"], on=["study_id", "level", "laterality"])

df.to_csv("../../data/train_gt_sag_subarticular_gt_kfold.csv", index=False)


