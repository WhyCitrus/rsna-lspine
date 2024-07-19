import cv2
import numpy as np
import os
import pandas as pd

from tqdm import tqdm


df = pd.read_csv("../../data/train_label_coordinates.csv")
spinal_df = df.loc[df.condition.apply(lambda x: "Spinal" in x)]

meta_df = pd.read_csv("../../data/dicom_metadata.csv")
meta_df = meta_df.loc[meta_df.series_id.isin(spinal_df.series_id.tolist())]
instance_to_position_index_dict = {}
for series_id, series_df in meta_df.groupby("series_id"):
	series_df = series_df.sort_values("ImagePositionPatient0", ascending=True)
	series_df["position_index"] = np.arange(len(series_df))
	instance_to_position_index_dict.update({f"{series_id}_{row.instance_number}": row.position_index for row in series_df.itertuples()})

levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
col_names = [f"rt_{_}" for _ in levels] + [f"lt_{_}" for _ in levels]

level_to_pixel_value = {"L1/L2": 175, "L2/L3": 195, "L3/L4": 215, "L4/L5": 235, "L5/S1": 255}
save_dir = "../../data/train_spinal_segmentation_maps_smaller/"
os.makedirs(save_dir, exist_ok=True)

for series_id, series_df in tqdm(spinal_df.groupby("series_id"), total=len(spinal_df.series_id.unique())):
	# skip if not full complement of right and left for each level
	if len(np.unique(series_df.condition + series_df.level)) != 5:
		continue
	# corresponding filepath of the image
	series_df["filepath"] = [f"{row.study_id}/{row.series_id}/IM{instance_to_position_index_dict[str(row.series_id) + '_' + str(row.instance_number)]:06d}_INST{row.instance_number:06d}.png" for row in series_df.itertuples()]
	series_meta = meta_df.loc[meta_df.series_id == series_id]
	rows, cols = series_meta.rows.iloc[0], series_meta.cols.iloc[0]
	tmp_mask = np.zeros((rows, cols, len(series_meta) + 1))
	for instance_number, instance_df in series_df.groupby("instance_number"):
		tmp_mask_inst = np.zeros((rows, cols))
		for inst_row_idx, inst_row in instance_df.iterrows():
			tmp_mask_inst = cv2.ellipse(tmp_mask_inst.copy(), (round(inst_row.x), round(inst_row.y)), (int(0.0125 * cols), int(0.0125 * rows)), 0, 0, 360, level_to_pixel_value[inst_row.level], -1)
			tmp_mask[:, :, inst_row.instance_number] = tmp_mask_inst
	tmp_mask_copy = np.zeros_like(tmp_mask)
	for ch in range(1, tmp_mask.shape[2]):
		if ch == tmp_mask.shape[2] - 1:
			tmp_mask_copy[:, :, -1] = tmp_mask[:, :, -1] + tmp_mask[:, :, -2] * 0.5
		else:
			tmp_mask_copy[:, :, ch] = tmp_mask[:, :, ch] + 0.5 * tmp_mask[:, :, ch + 1] + 0.5 * tmp_mask[:, :, ch - 1]
	tmp_mask_copy = tmp_mask_copy.astype("uint8")
	# ch corresponds to instance number
	for ch in range(tmp_mask_copy.shape[2]):
		if tmp_mask_copy[..., ch].sum() == 0:
			continue
		tmp_save_dir = os.path.join(save_dir, str(series_df.study_id.values[0]), str(series_id))
		os.makedirs(tmp_save_dir, exist_ok=True)
		try:
			tmp_filepath = f"IM{instance_to_position_index_dict[f'{series_id}_{ch}']:06d}_INST{ch:06d}.png"
			status = cv2.imwrite(os.path.join(tmp_save_dir, tmp_filepath), tmp_mask_copy[..., ch])
		except KeyError as e:
			print(series_id, e)
			continue
