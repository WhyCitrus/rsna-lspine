import cv2
import glob
import numpy as np
import os
import pandas as pd


df = pd.read_csv("../../data/train_label_coordinates.csv")
df = df.loc[df.condition.apply(lambda x: "Subarticular" in x)]
df["level"] = df["level"].apply(lambda x: x.replace("/", "_"))
df["laterality"] = df.condition.apply(lambda x: x.split(" ")[0][:1] + "T")

for row in df.itertuples():
	filepath = os.path.join("../../data/train_pngs", str(row.study_id), str(row.series_id), f"IM{row.instance_number:06d}.png")
	img = cv2.imread(filepath, 0)
	h, w = int(0.15 * img.shape[0]), int(0.15 * img.shape[1])
	xc, yc = int(row.x), int(row.y)
	x1, y1 = xc - w // 2, yc - h // 2
	x2, y2 = x1 + w, y1 + h
	x1, y1 = max(0, x1), max(0, y1)
	x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
	cropped = img[y1:y2, x1:x2]
	savepath = os.path.join("../../data/train_subarticular_patches_using_ground_truth_coords/", str(row.study_id), f"{row.laterality}_{row.level}.png")
	os.makedirs(os.path.dirname(savepath), exist_ok=True)
	status = cv2.imwrite(savepath, cropped)


train_df = pd.read_csv("../../data/train.csv")
train_df = train_df[["study_id"] + [c for c in train_df.columns if "subarticular" in c]]

df_list = []
for c in train_df.columns:
	if c == "study_id":
		continue
	tmp_df = train_df[["study_id", c]].copy()
	tmp_df["laterality"] = c.split("_")[0][0].upper()
	tmp_df["level"] = "_".join(c.split("_")[-2:]).upper()
	tmp_df.columns = ["study_id", "grade", "laterality", "level"]
	df_list.append(tmp_df)

train_df = pd.concat(df_list)
labels_map = {
	"Normal/Mild": 0, "Moderate": 1, "Severe": 2
}
train_df = train_df.loc[~train_df.grade.isna()]
train_df["grade"] = train_df["grade"].map(labels_map).astype("int")
train_df["normal_mild"] = (train_df.grade == 0).astype("int")
train_df["moderate"] = (train_df.grade == 1).astype("int")
train_df["severe"] = (train_df.grade == 2).astype("int")

folds_df = pd.read_csv("../../data/folds_cv5.csv")

images = glob.glob("../../data/train_subarticular_patches_using_ground_truth_coords/*/*.png")

df = pd.DataFrame({"filepath": images})
df["filepath"] = df.filepath.apply(lambda x: x.replace("../../data/train_subarticular_patches_using_ground_truth_coords/", ""))
df["study_id"] = df.filepath.apply(lambda x: x.split("/")[0]).astype("int")
df["laterality"] = df.filepath.apply(lambda x: os.path.basename(x).split("_")[0][:1])
df["level"] = df.filepath.apply(lambda x: os.path.basename(x)[3:].replace(".png", ""))

df = df.merge(train_df, on=["study_id", "laterality", "level"])
df = df.merge(folds_df, on="study_id")

df.to_csv("../../data/train_subarticular_crops_using_ground_truth_coords.csv", index=False)
