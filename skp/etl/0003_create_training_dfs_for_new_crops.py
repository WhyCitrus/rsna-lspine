import glob
import os
import pandas as pd


def create_training_df(image_dir, condition):
	images = glob.glob(os.path.join(image_dir, "*.png"))
	df = pd.DataFrame({"filepath": images})
	df["filepath"] = df.filepath.map(os.path.basename)
	df["study_id"] = df.filepath.apply(lambda x: x.split("_")[0]).astype("int")
	if "canal" in image_dir:
		df["level"] = df.filepath.apply(lambda x: "_".join(x.replace(".png", "").split("_")[1:3]))
	else:
		df["level"] = df.filepath.apply(lambda x: "_".join(x.replace(".png", "").split("_")[2:4]))
	df["laterality"] = df.filepath.apply(lambda x: x.split("_")[1][:1]) if "canal" not in image_dir else None
	df = df.merge(folds_df, on="study_id")
	merge_cols = ["study_id", "level"]
	if "canal" not in image_dir:
		merge_cols.append("laterality")
	else:
		del df["laterality"]
	df = df.merge(train_df.loc[train_df.condition == condition], on=merge_cols)
	return df


folds_df = pd.read_csv("../../data/folds_cv5.csv")
train_df = pd.read_csv("../../data/train_narrow.csv")

gen_spinal = create_training_df("../../data/train_crops_generated_by_3d_models/canal/", "spinal")
gen_foraminal = create_training_df("../../data/train_crops_generated_by_3d_models/foramina/", "foraminal")
gen_subarticular = create_training_df("../../data/train_crops_generated_by_3d_models/subarticular/", "subarticular")

gen_spinal.to_csv("../../data/train_gen_3d_spinal_crops_kfold.csv", index=False)
gen_foraminal.to_csv("../../data/train_gen_3d_foraminal_crops_kfold.csv", index=False)
gen_subarticular.to_csv("../../data/train_gen_3d_subarticular_crops_kfold.csv", index=False)
