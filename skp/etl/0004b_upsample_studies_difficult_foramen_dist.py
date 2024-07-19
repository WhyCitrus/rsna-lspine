import pandas as pd


df = pd.read_csv("../../data/train_foramen_distance_each_level_with_foramen_coords.csv")

with open("../../data/difficult_foramina_dist_predict_studies.txt", "r") as f:
	difficult_studies = [int(_.strip()) for _ in f.readlines()]

difficult_studies = df.loc[df.study_id.isin(difficult_studies)]
# upsample
df = pd.concat([df] + [difficult_studies] * 3)

df.to_csv("../../data/train_foramen_dist_each_level_with_coords_upsample_hard_cases.csv", index=False)
