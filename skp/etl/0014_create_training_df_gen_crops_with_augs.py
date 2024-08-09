import glob
import os
import pandas as pd


train_df = pd.read_csv("../../data/train_narrow.csv")
folds_df = pd.read_csv("../../data/folds_cv5.csv")

foramen_crops = glob.glob("../../data/train_generated_crops_with_augs_dist_coord_proba/foramina/*.png")
foramen_df = pd.DataFrame({"filepath": foramen_crops})
foramen_df["filepath"] = foramen_df.filepath.apply(lambda x: os.path.basename(x))
foramen_df["study_id"] = foramen_df.filepath.apply(lambda x: x.split("_")[0]).astype("int")
foramen_df["series_id"] = foramen_df.filepath.apply(lambda x: x.split("_")[1]).astype("int")
foramen_df["laterality"] = foramen_df.filepath.apply(lambda x: x.split("_")[2][:1])
foramen_df["level"] = foramen_df.filepath.apply(lambda x: "_".join(x.split("_")[3:5]))
foramen_df["unique_identifier"] = foramen_df.study_id.astype("str") + "_" + foramen_df.laterality + "_" + foramen_df.level
foramen_df["unique_id"] = pd.Categorical(foramen_df.unique_identifier).codes
foramen_df = foramen_df.merge(folds_df, on="study_id")
foramen_df = foramen_df.merge(train_df.loc[train_df.condition == "foraminal"], on=["study_id", "level", "laterality"])
foramen_df.to_csv("../../data/train_gen_foramen_crops_with_augs.csv", index=False)
foramen_df.loc[foramen_df.filepath.apply(lambda x: x.endswith("000.png"))].to_csv("../../data/train_gen_foramen_crops_without_augs.csv", index=False)

# concat_foramen_df = []
# for unique_id, _df in foramen_df.groupby("unique_id"):
# 	assert len(_df) == 27
# 	_df = _df.sort_values("filepath")
# 	concat_filepath = ",".join(_df.filepath.tolist())
# 	_df["filepath"] = concat_filepath
# 	_df = _df.drop_duplicates()
# 	concat_foramen_df.append(_df)

# concat_foramen_df = pd.concat(concat_foramen_df)
# concat_foramen_df.to_csv("../../data/train_gen_concat_foramen_crops_with_augs.csv", index=False)


spinal_crops = glob.glob("../../data/train_generated_crops_with_augs_dist_coord_proba/spinal/*.png")
spinal_df = pd.DataFrame({"filepath": spinal_crops})
spinal_df["filepath"] = spinal_df.filepath.apply(lambda x: os.path.basename(x))
spinal_df["study_id"] = spinal_df.filepath.apply(lambda x: x.split("_")[0]).astype("int")
spinal_df["series_id"] = spinal_df.filepath.apply(lambda x: x.split("_")[1]).astype("int")
spinal_df["level"] = spinal_df.filepath.apply(lambda x: "_".join(x.split("_")[2:4]))
spinal_df["unique_identifier"] = spinal_df.study_id.astype("str") + "_" + spinal_df.level
spinal_df["unique_id"] = pd.Categorical(spinal_df.unique_identifier).codes
spinal_df = spinal_df.merge(folds_df, on="study_id")
spinal_df = spinal_df.merge(train_df.loc[train_df.condition == "spinal"], on=["study_id", "level"])
spinal_df.to_csv("../../data/train_gen_spinal_crops_with_augs.csv", index=False)
spinal_df.loc[spinal_df.filepath.apply(lambda x: x.endswith("000.png"))].to_csv("../../data/train_gen_spinal_crops_without_augs.csv", index=False)


subarticular_crops = glob.glob("../../data/train_generated_crops_with_augs_dist_coord_proba/subarticular/*.png")
subarticular_df = pd.DataFrame({"filepath": subarticular_crops})
subarticular_df["filepath"] = subarticular_df.filepath.apply(lambda x: os.path.basename(x))
subarticular_df["study_id"] = subarticular_df.filepath.apply(lambda x: x.split("_")[0]).astype("int")
subarticular_df["series_id"] = subarticular_df.filepath.apply(lambda x: x.split("_")[1]).astype("int")
subarticular_df["laterality"] = subarticular_df.filepath.apply(lambda x: x.split("_")[2][:1])
subarticular_df["level"] = subarticular_df.filepath.apply(lambda x: "_".join(x.split("_")[3:5]))
subarticular_df["unique_identifier"] = subarticular_df.study_id.astype("str") + "_" + subarticular_df.laterality + "_" + subarticular_df.level
subarticular_df["unique_id"] = pd.Categorical(subarticular_df.unique_identifier).codes
subarticular_df = subarticular_df.merge(folds_df, on="study_id")
subarticular_df = subarticular_df.merge(train_df.loc[train_df.condition == "subarticular"], on=["study_id", "level", "laterality"])
subarticular_df.to_csv("../../data/train_gen_subarticular_crops_with_augs.csv", index=False)
subarticular_df.loc[subarticular_df.filepath.apply(lambda x: x.endswith("000.png"))].to_csv("../../data/train_gen_subarticular_crops_without_augs.csv", index=False)
