import glob
import pandas as pd


df = pd.read_csv("../../data/train_gt_subarticular_crops_kfold.csv")
rdf = df.loc[df.laterality == "R"][["study_id", "level", "normal_mild", "moderate", "severe"]]
ldf = df.loc[df.laterality == "L"][["study_id", "level", "normal_mild", "moderate", "severe"]]
rdf.columns = ["study_id", "level", "rt_normal_mild", "rt_moderate", "rt_severe"]
ldf.columns = ["study_id", "level", "lt_normal_mild", "lt_moderate", "lt_severe"]
bdf = rdf.merge(ldf, on=["study_id", "level"])

images = glob.glob("../../data/train_subarticular_full_slices/*.png")
image_df = pd.DataFrame({"filepath": images})
image_df["filepath"] = image_df.filepath.apply(lambda x: x.replace("../../data/train_subarticular_full_slices/", ""))
image_df["study_id"] = image_df.filepath.apply(lambda x: x.split("_")[0]).astype("int")
image_df["level"] = image_df.filepath.apply(lambda x: x.replace(".png", "")[-5:])
df = image_df.merge(bdf, on=["study_id", "level"])
df["sample_weight"] = 1
df.loc[df.rt_moderate + df.lt_moderate > 0, "sample_weight"] = 2
df.loc[df.rt_severe + df.lt_severe > 0, "sample_weight"] = 4

folds_df = pd.read_csv("../../data/folds_cv5.csv")
df = df.merge(folds_df, on="study_id")
df.to_csv("../../data/train_subarticular_full_slices_kfold.csv", index=False)
