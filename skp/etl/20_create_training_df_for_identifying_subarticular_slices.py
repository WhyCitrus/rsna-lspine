import glob
import pandas as pd


df = pd.read_csv("../../data/train_label_coordinates.csv")
df = df.loc[df.condition.apply(lambda x: "Subarticular" in x)]
df["filepath"] = df.study_id.astype(str) + "/" + df.series_id.astype(str) + "/" + df.instance_number.apply(lambda x: f"IM{x:06d}.png")

all_images = glob.glob("../../data/train_pngs/*/*/*.png")

image_df = pd.DataFrame({"filepath": all_images})
image_df["filepath"] = image_df.filepath.apply(lambda x: x.replace("../../data/train_pngs/", ""))
folds_df = pd.read_csv("../../data/folds_cv5.csv")

image_df["study_id"] = image_df.filepath.apply(lambda x: x.split("/")[0]).astype("int")
image_df["series_id"] = image_df.filepath.apply(lambda x: x.split("/")[1]).astype("int")
image_df["target"] = 0
image_df.loc[image_df.filepath.isin(df.filepath.tolist()), "target"] = 1
print(image_df.target.value_counts())
image_df = image_df.merge(folds_df, on="study_id")
image_df

image_df.to_csv("../../data/train_identify_subarticular_slices_level_naive.csv", index=False)
