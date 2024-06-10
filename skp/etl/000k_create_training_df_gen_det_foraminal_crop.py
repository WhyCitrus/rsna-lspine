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

gen_foraminal = create_training_df("../../data/train_generated_crops_using_detection/foraminal/", "foraminal")
gen_foraminal.to_csv("../../data/train_gen_det_foraminal_crops_kfold.csv", index=False)
