import cv2
import numpy as np
import os
import pandas as pd

from tqdm import tqdm


def crop_square_around_center(img, xc, yc, size_factor=0.15):
    h, w = size_factor * img.shape[0], size_factor * img.shape[1]
    x1, y1 = xc - w / 2, yc - h / 2
    x2, y2 = x1 + w, y1 + h
    x1, y1, x2, y2 = [int(_) for _ in [x1, y1, x2, y2]]
    return img[y1:y2, x1:x2]


coord_df = pd.read_csv("../../data/train_label_coordinates.csv")
coord_df = coord_df.loc[coord_df.condition.apply(lambda x: "Foraminal" in x)]

# The following studies are incorrectly labeled (e.g., left and right sides are swapped or one side is labeled left and right)
incorrectly_labeled = [
    3742728457, 364930790, 2410494888, 2135829458, 1879696087, 3156269631, 1085426528, 3781188430, 3495818564,
    1395773918, 757619082, 2626030939, 2530679352, 2388577668, 1647904243, 796739553, 2662989538
]

coord_df = coord_df.loc[~coord_df.study_id.isin(incorrectly_labeled)]
coord_df = pd.concat([
    series_df for series_id, series_df in coord_df.groupby("series_id") if len(np.unique(series_df.condition + series_df.level)) == 10
])

meta_df = pd.read_csv("../../data/dicom_metadata.csv")
meta_df = meta_df.loc[meta_df.series_id.isin(coord_df.series_id.tolist())]

instance_to_position_index_dict = {}
for series_id, series_df in meta_df.groupby("series_id"):
    series_df = series_df.sort_values("ImagePositionPatient0", ascending=True)
    series_df["position_index"] = np.arange(len(series_df))
    instance_to_position_index_dict.update({f"{series_id}_{row.instance_number}": row.position_index for row in series_df.itertuples()})

meta_df["series_instance"] = meta_df.series_id.astype("str") + "_" + meta_df.instance_number.astype("str")
meta_df["position_index"] = meta_df.series_instance.map(instance_to_position_index_dict)

coord_df["series_instance"] = coord_df.series_id.astype("str") + "_" + coord_df.instance_number.astype("str")
coord_df["position_index"] = coord_df.series_instance.map(instance_to_position_index_dict)

levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]

image_dir = "../../data/train_pngs_3ch/"
save_dir = "../../data/train_foramen_dist_predict_crops"
os.makedirs(save_dir, exist_ok=True)

df_list = []
for series_id, series_df in tqdm(coord_df.groupby("series_id"), total=len(coord_df.series_id.unique())):
    series_meta = meta_df.loc[meta_df.series_id == series_id]
    series_meta = series_meta.sort_values("ImagePositionPatient0")
    series_df = series_df.merge(series_meta, on=list(set(series_meta.columns) & set(series_df.columns)))
    series_df = series_df.sort_values(["condition", "level"]).reset_index(drop=True)
    lt_coords = {levels[row_idx]: (row.ImagePositionPatient0, row.x, row.y) for row_idx, row in series_df.iloc[:5].iterrows()}
    rt_coords = {levels[row_idx-5]: (row.ImagePositionPatient0, row.x, row.y) for row_idx, row in series_df.iloc[5:].iterrows()}
    for lvl in levels:
        series_meta[f"lt_{lvl}_dist"] = [row.ImagePositionPatient0 - lt_coords[lvl][0] for row in series_meta.itertuples()]
        series_meta[f"rt_{lvl}_dist"] = [row.ImagePositionPatient0 - rt_coords[lvl][0] for row in series_meta.itertuples()]
        series_meta[f"{lvl}_dist"] = np.stack([series_meta[f"lt_{lvl}_dist"].abs().values, series_meta[f"rt_{lvl}_dist"].abs().values], axis=1).min(1)
        series_meta[f"{lvl}_which_side"] = (series_meta[f"lt_{lvl}_dist"].abs().values > series_meta[f"rt_{lvl}_dist"].abs().values).astype("int")
    df_list.append(series_meta)
    for row_idx, row in series_meta.iterrows():
        img = cv2.imread(os.path.join(image_dir, str(row.study_id), str(row.series_id), f"IM{row.position_index:06d}_INST{row.instance_number:06d}.png"))
        offset_x, offset_y = int(0.0175 * img.shape[1]), int(0.0175 * img.shape[0])
        for lvl in levels:
            tmp_coords = rt_coords if row[f"{lvl}_which_side"] == 1 else lt_coords
            x, y = tmp_coords[lvl][1], tmp_coords[lvl][2]
            xs = [x, x - offset_x, x + offset_x]
            ys = [y, y - offset_y, y + offset_y]
            crops = []
            for xi in xs:
                for yi in ys:
                    crops.append(crop_square_around_center(img, xi, yi, size_factor=0.15))
            for crop_idx, each_crop in enumerate(crops):
                tmp_filepath = os.path.join(save_dir, f"{row.study_id}_{row.series_id}_{lvl.upper()}_IM{row.position_index:06d}_INST{row.instance_number:06d}_CROP{crop_idx:06d}.png")
                status = cv2.imwrite(tmp_filepath, each_crop)

folds_df = pd.read_csv("../../data/folds_cv5.csv")
df = pd.concat(df_list)
df = df.merge(folds_df, on="study_id")
df.to_csv("../../data/train_foramen_dist_predict_crops.csv", index=False)
