import pandas as pd


df = pd.read_csv("../../data/train_gen_subarticular_crops_with_augs.csv")

lt_df = df.loc[df.laterality == "L"]
rt_df = df.loc[df.laterality == "R"]

lt_df["filepath"] = lt_df.filepath.apply(lambda x: x.replace("LT_", ""))
rt_df["filepath"] = rt_df.filepath.apply(lambda x: x.replace("RT_", ""))

lt_df_cols = list(lt_df.columns)
rt_df_cols = list(rt_df.columns)

lt_df_cols[-4:-1] = [f"lt_{_}" for _ in lt_df_cols[-4:-1]]
rt_df_cols[-4:-1] = [f"rt_{_}" for _ in rt_df_cols[-4:-1]]

lt_df.columns = lt_df_cols
rt_df.columns = rt_df_cols

df = lt_df.merge(rt_df[["filepath", "rt_normal_mild", "rt_moderate", "rt_severe"]], on="filepath")
df.to_csv("../../data/train_gen_bilateral_subarticular_crops_with_augs.csv", index=False)
