import pandas as pd


df = pd.read_csv("../../data/train.csv")
foramina_df = df[[c for c in df.columns if "subarticular" not in c and "canal" not in c]]
series_descs = pd.read_csv("../../data/train_series_descriptions.csv")

foramina_df = foramina_df.merge(series_descs, on="study_id")
foramina_df = foramina_df.loc[foramina_df.series_description == "Sagittal T1"]

for c in foramina_df.columns:
    if "foraminal" in c:
        print(c, foramina_df[c].isna().sum())