import pandas as pd


canal_df = pd.read_csv("../../data/train_sagittal_canal_crops.csv")
canal_df["canal"] = 1
canal_df["subarticular"] = 0
canal_df["foramen"] = 0
canal_df["filepath"] = "train_sagittal_canal_crops_3ch/" + canal_df.filepath

# Don't do separate areas for laterality since we are flipping
subart_df = pd.read_csv("../../data/train_subarticular_crops.csv")
subart_df["canal"] = 0
subart_df["subarticular"] = 1
subart_df["foramen"] = 0
subart_df["filepath"] = "train_axial_unilateral_subarticular_crops_3ch/" + subart_df.filepath

foramen_df = pd.read_csv("../../data/train_foramina_crops.csv")
foramen_df["canal"] = 0
foramen_df["subarticular"] = 0
foramen_df["foramen"] = 1
foramen_df["filepath"] = "train_foramina_crops_3ch/" + foramen_df.filepath

# Repeat canal_df twice since it is weighted 2x in metric since subarticular zones and foramina are bilateral
# but canal is only unilateral
df = pd.concat([canal_df, canal_df, subart_df, foramen_df])
df.to_csv("../../data/train_combined_areas_crops.csv", index=False)
