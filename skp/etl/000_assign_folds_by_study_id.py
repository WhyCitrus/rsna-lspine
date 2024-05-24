from utils import create_double_cv


df = pd.read_csv("../../data/train.csv")
df = create_double_cv(df[["study_id"]], "study_id", 5, 5)
df
df.to_csv("../../data/folds_cv5.csv", index=False)