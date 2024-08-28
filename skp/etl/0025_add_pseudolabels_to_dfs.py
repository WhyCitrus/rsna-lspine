import pandas as pd


pseudo = pd.read_csv("../../data/ns_oofs_v4.csv")
pseudo = pseudo.loc[pseudo.row_id.apply(lambda x: "foraminal" in x)]
pseudo["study_id"] = pseudo.row_id.apply(lambda x: x.split("_")[0]).astype("int")
pseudo["laterality"] = pseudo.row_id.apply(lambda x: x.split("_")[1][:1].upper())
pseudo["level"] = pseudo.row_id.apply(lambda x: "_".join(x.split("_")[-2:]).upper())

df = pd.read_csv("../../data/train_gt_foraminal_with_augs_kfold.csv")
df = df.merge(pseudo, on=["study_id", "level", "laterality"], suffixes=["", "_pseudo"])
df.to_csv("../../data/train_gt_foraminal_with_augs_pseudo_kfold.csv", index=False)

