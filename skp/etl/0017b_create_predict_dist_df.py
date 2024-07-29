import glob
import os
import pandas as pd


df = pd.read_csv("../../data/train_foramen_dist_predict_crops.csv")
crops = glob.glob("../../data/train_foramen_dist_predict_crops/*png")
crop_df = pd.DataFrame({"filepath": crops})
crop_df["filepath"] = crop_df.filepath.apply(lambda x: os.path.basename(x))
crop_df["study_id"] = crop_df.filepath.apply(lambda x: x.split("_")[0]).astype("int")
crop_df["series_id"] = crop_df.filepath.apply(lambda x: x.split("_")[1]).astype("int")
crop_df["level"] = crop_df.filepath.apply(lambda x: "_".join(x.split("_")[2:4]))
crop_df["position_index"] = crop_df.filepath.apply(lambda x: x.split("_")[4].replace("IM", "")).astype("int")
crop_df["instance_number"] = crop_df.filepath.apply(lambda x: x.split("_")[5].replace("INST", "")).astype("int")
crop_df["crop_index"] = crop_df.filepath.apply(lambda x: x.split("_")[-1].replace("CROP", "").replace(".png", "")).astype("int")

levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
df_list = []
for series_id, series_df in df.groupby("series_id"):
	series_df_list = [series_df.copy()] * 5
	tmp_df_list = []
	for tmp_df, lvl in zip(series_df_list, levels):
		tmp_df_copy = tmp_df.copy()
		tmp_df_copy["dist"] = tmp_df_copy[f"{lvl}_dist"]
		tmp_df_copy["level"] = lvl.upper()
		tmp_df_list.append(tmp_df_copy.copy())
	tmp_df_copy = pd.concat(tmp_df_list)
	df_list.append(tmp_df_copy)

dist_df = pd.concat(df_list)
del dist_df["filepath"]
crop_df = crop_df.merge(dist_df, on=["study_id", "series_id", "level", "position_index", "instance_number"])
crop_df.to_csv("../../data/train_foramen_dist_predict_crops_kfold.csv", index=False)
