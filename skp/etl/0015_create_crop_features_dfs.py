import os
import pandas as pd


foramen_df = pd.read_csv("../../data/train_gen_foramen_crops_with_augs.csv")
foramen_df["filepath"] = foramen_df.study_id.astype("str") + "_" + foramen_df.series_id.astype("str") + "_" + foramen_df.laterality + "_" + foramen_df.level + ".npy"
foramen_df.drop_duplicates().to_csv("../../data/train_foramen_crop_features.csv", index=False)

spinal_df.to_csv("../../data/train_gen_spinal_crops_with_augs.csv", index=False)
subarticular_df.to_csv("../../data/train_gen_subarticular_crops_with_augs.csv", index=False)
