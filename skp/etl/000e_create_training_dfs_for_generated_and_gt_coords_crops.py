import glob
import os
import pandas as pd


def create_training_df(image_dir, condition):
	images = glob.glob(os.path.join(image_dir, "*.png"))
	df = pd.DataFrame({"filepath": images})
	df["filepath"] = df.filepath.map(os.path.basename)
	df["study_id"] = df.filepath.apply(lambda x: x.split("_")[0]).astype("int")
	df["level"] = df.filepath.apply(lambda x: x.replace(".png", "")[-5:])
	df["laterality"] = df.filepath.apply(lambda x: x.split("_")[-3]) if "spinal" not in image_dir else None
	df = df.merge(folds_df, on="study_id")
	merge_cols = ["study_id", "level"]
	if "spinal" not in image_dir:
		merge_cols.append("laterality")
	else:
		del df["laterality"]
	df = df.merge(train_df.loc[train_df.condition == condition], on=merge_cols)
	return df


folds_df = pd.read_csv("../../data/folds_cv5.csv")
train_df = pd.read_csv("../../data/train_wide.csv")

gen_spinal = create_training_df("../../data/train_generated_crops/spinal/", "spinal")
gen_foraminal = create_training_df("../../data/train_generated_crops/foraminal/", "foraminal")
gen_subarticular = create_training_df("../../data/train_generated_crops/subarticular/", "subarticular")

gt_spinal = create_training_df("../../data/train_crops_gt_coords/spinal/", "spinal")
gt_foraminal = create_training_df("../../data/train_crops_gt_coords/foraminal/", "foraminal")
gt_subarticular = create_training_df("../../data/train_crops_gt_coords/subarticular/", "subarticular")

gen_spinal.shape
gen_foraminal.shape
gen_subarticular.shape
gt_spinal.shape
gt_foraminal.shape
gt_subarticular.shape

gen_spinal.to_csv("../../data/train_gen_spinal_crops_kfold.csv", index=False)
gen_foraminal.to_csv("../../data/train_gen_foraminal_crops_kfold.csv", index=False)
gen_subarticular.to_csv("../../data/train_gen_subarticular_crops_kfold.csv", index=False)

gt_spinal.to_csv("../../data/train_gt_spinal_crops_kfold.csv", index=False)
gt_foraminal.to_csv("../../data/train_gt_foraminal_crops_kfold.csv", index=False)
gt_subarticular.to_csv("../../data/train_gt_subarticular_crops_kfold.csv", index=False)

gen_spinal["filepath"] = "spinal/" + gen_spinal.filepath
gen_foraminal["filepath"] = "foraminal/" + gen_foraminal.filepath
gen_subarticular["filepath"] = "subarticular/" + gen_subarticular.filepath
gen_spinal["sample_weight"] *= 2

gen_all = pd.concat([gen_spinal, gen_foraminal, gen_subarticular])
gen_all.to_csv("../../data/train_gen_all_crops_kfold.csv", index=False)

gen_spinal["filepath"] = "train_generated_crops/spinal/" + gen_spinal.filepath
gen_foraminal["filepath"] = "train_generated_crops/foraminal/" + gen_foraminal.filepath
gen_subarticular["filepath"] = "train_generated_crops/subarticular/" + gen_subarticular.filepath

gt_spinal["filepath"] = "train_crops_gt_coords/spinal/" + gt_spinal.filepath
gt_foraminal["filepath"] = "train_crops_gt_coords/foraminal/" + gt_foraminal.filepath
gt_subarticular["filepath"] = "train_crops_gt_coords/subarticular/" + gt_subarticular.filepath

gen_spinal["ignore_during_val"] = 0
gen_foraminal["ignore_during_val"] = 0
gen_subarticular["ignore_during_val"] = 0

gt_spinal["ignore_during_val"] = 1
gt_foraminal["ignore_during_val"] = 1
gt_subarticular["ignore_during_val"] = 1

combo_spinal = pd.concat([gen_spinal, gt_spinal])
combo_foraminal = pd.concat([gen_foraminal, gt_foraminal])
combo_subarticular = pd.concat([gen_subarticular, gt_subarticular])

combo_spinal.to_csv("../../data/train_gen_plus_gt_spinal_crops_kfold.csv", index=False)
combo_foraminal.to_csv("../../data/train_gen_plus_gt_foraminal_crops_kfold.csv", index=False)
combo_subarticular.to_csv("../../data/train_gen_plus_gt_subarticular_crops_kfold.csv", index=False)
