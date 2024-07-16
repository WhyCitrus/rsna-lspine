import glob
import os
import pandas as pd


def create_training_df(image_dir, condition):
	images = glob.glob(os.path.join(image_dir, "*.png"))
	df = pd.DataFrame({"filepath": images})
	df["filepath"] = df.filepath.map(os.path.basename)
	df["study_id"] = df.filepath.apply(lambda x: x.split("_")[0]).astype("int")
	df["level"] = df.filepath.apply(lambda x: x.replace(".png", "")[-5:])
	df["laterality"] = df.filepath.apply(lambda x: x.split("_")[-3][:1]) if "canal" not in image_dir else None
	df = df.merge(folds_df, on="study_id")
	merge_cols = ["study_id", "level"]
	if "canal" not in image_dir:
		merge_cols.append("laterality")
	else:
		del df["laterality"]
	df = df.merge(train_df.loc[train_df.condition == condition], on=merge_cols)
	df["series_folder"] = df.filepath.apply(lambda x: x.replace(".png", ""))
	return df

folds_df = pd.read_csv("../../data/folds_cv5.csv")
train_df = pd.read_csv("../../data/train_narrow.csv")

gen_spinal = create_training_df("../../data/train_generated_crops_using_detection2/canal/", "spinal")
gen_foraminal = create_training_df("../../data/train_generated_crops_using_detection2/foramina/", "foraminal")
gen_subarticular = create_training_df("../../data/train_generated_crops_using_detection2/subarticular/", "subarticular")

gen_spinal.shape
gen_foraminal.shape
gen_subarticular.shape

gen_spinal.to_csv("../../data/train_gen_det2_spinal_crops_kfold.csv", index=False)
gen_foraminal.to_csv("../../data/train_gen_det2_foraminal_crops_kfold.csv", index=False)
gen_subarticular.to_csv("../../data/train_gen_det2_subarticular_crops_kfold.csv", index=False)

gen_spinal["filepath"] = "canal/" + gen_spinal.filepath
gen_foraminal["filepath"] = "foramina/" + gen_foraminal.filepath
gen_subarticular["filepath"] = "subarticular/" + gen_subarticular.filepath

gen_spinal["series_folder"] = "canal/" + gen_spinal.series_folder
gen_foraminal["series_folder"] = "foramina/" + gen_foraminal.series_folder
gen_subarticular["series_folder"] = "subarticular/" + gen_subarticular.series_folder

gen_spinal["sample_weight"] *= 2

gen_all = pd.concat([gen_spinal, gen_foraminal, gen_subarticular])
gen_all.to_csv("../../data/train_gen_det2_all_crops_kfold.csv", index=False)
