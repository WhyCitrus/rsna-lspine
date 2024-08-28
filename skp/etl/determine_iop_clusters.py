import numpy as np
import pandas as pd
import pydicom

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm


meta_df = pd.read_csv("../../data/dicom_metadata.csv")
desc_df = pd.read_csv("../../data/train_series_descriptions.csv")
desc_df = desc_df.loc[desc_df.series_description == "Axial T2"]
meta_df = meta_df.loc[meta_df.series_id.isin(desc_df.series_id.tolist())]

df_list = []
silhouette_score_dict = {}
for series_id, series_df in tqdm(meta_df.groupby("series_id"), total=len(meta_df.series_id.unique())):
	series_df = series_df.sort_values("ImagePositionPatient2")
	dicoms = [pydicom.dcmread(f"../../data/train_images/{row.study_id}/{row.series_id}/{row.filename}", stop_before_pixels=True) for row in series_df.itertuples()]
	iop = [d.ImageOrientationPatient for d in dicoms]
	iop = np.stack(iop)
	km = KMeans(n_clusters=5, random_state=0, n_init="auto")
	cluster_labels = km.fit_predict(iop)
	try:
		score = silhouette_score(iop, cluster_labels)
	except ValueError:
		score = 1
	silhouette_score_dict[series_id] = score
	series_df["cluster_label"] = cluster_labels
	series_df["ImageOrientationPatient0"] = iop[:, 0]
	series_df["ImageOrientationPatient1"] = iop[:, 1]
	series_df["ImageOrientationPatient2"] = iop[:, 2]
	df_list.append(series_df)

dfs = pd.concat(df_list)

for series_id, series_df in dfs.groupby("series_id"):
	for cluster in series_df.groupby("cluster_labels"):
