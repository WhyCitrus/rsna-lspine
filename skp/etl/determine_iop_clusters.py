import numpy as np
import pandas as pd
import pydicom

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from tqdm import tqdm


meta_df = pd.read_csv("../../data/dicom_metadata.csv")
desc_df = pd.read_csv("../../data/train_series_descriptions.csv")
desc_df = desc_df.loc[desc_df.series_description == "Axial T2"]
meta_df = meta_df.loc[meta_df.series_id.isin(desc_df.series_id.tolist())]

df_list = []
silhouette_score_dict = {}
db = DBSCAN(eps=0.001, min_samples=2)
for series_id, series_df in tqdm(meta_df.groupby("series_id"), total=len(meta_df.series_id.unique())):
	series_df = series_df.sort_values("ImagePositionPatient2")
	dicoms = [pydicom.dcmread(f"../../data/train_images/{row.study_id}/{row.series_id}/{row.filename}", stop_before_pixels=True) for row in series_df.itertuples()]
	iop = [d.ImageOrientationPatient for d in dicoms]
	iop = np.stack(iop)
	series_df["ImageOrientationPatient0"] = iop[:, 0]
	series_df["ImageOrientationPatient1"] = iop[:, 1]
	series_df["ImageOrientationPatient2"] = iop[:, 2]
	if np.unique(iop, axis=1).shape[0] == 1:
		silhouette_score[series_id] = 1
		series_df["cluster_labels"] = 0
	else:
		cluster_labels = db.fit_predict(iop)
		try:
			score = silhouette_score(iop, cluster_labels)
		except ValueError:
			score = 1
		silhouette_score_dict[series_id] = score
		series_df["cluster_label"] = cluster_labels
	df_list.append(series_df)

dfs = pd.concat(df_list)

{k: v for k, v in silhouette_score_dict.items() if v < 0.9}


is_sorted = lambda a: np.all(a[:-1] <= a[1:])

df_list = []
for series_id, series_df in dfs.groupby("series_id"):
	tmp_df_list = []
	tmp_centers = []
	for cluster, cluster_df in series_df.groupby("cluster_label"):
		cluster_df = cluster_df.sort_values("ImagePositionPatient2")
		tmp_df_list.append(cluster_df)
		tmp_centers.append(cluster_df.ImagePositionPatient2.max())
	sort_indices = np.argsort(tmp_centers)
	tmp_df_list = [tmp_df_list[idx] for idx in sort_indices]
	tmp_df = pd.concat(tmp_df_list)
	assert is_sorted(tmp_df.ImagePositionPatient2.values)


