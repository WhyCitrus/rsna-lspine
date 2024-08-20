import pandas as pd


df = pd.read_csv("../../data/train_gt_foraminal_with_augs_kfold.csv")
df["series_id"] = df.filepath.apply(lambda x: x.split("_")[1]).astype("int")
df["instance_number"] = df.filepath.apply(lambda x: x.split("_")[5].replace("INST", "")).astype("int")
df["series_level_instance"] = df.series_id.astype("str") + "-" + df.level + "-" + df.instance_number.astype("str")

coords_df = pd.read_csv("../../data/train_label_coordinates.csv")
coords_df = coords_df.loc[coords_df.condition.apply(lambda x: "Foraminal" in x)]
coords_df["series_level_instance"] = coords_df.series_id.astype("str") + "-" + coords_df.level.apply(lambda x: x.replace("/", "_")) + "-" + coords_df.instance_number.astype("str")

df["label"] = 0
df.loc[df.series_level_instance.isin(coords_df.series_level_instance.tolist()), "label"] = 1
df.label.value_counts()

df.to_csv("../../data/train_gt_foraminal_rank_crops_df.csv", index=False)
