import cv2
import glob
import os

from tqdm import tqdm


images = glob.glob("../../data/train_pngs_3ch/*/*/*.png")
save_dir = "../../data/train_pngs_1ch/"

for im in tqdm(images):
    img = cv2.imread(im)
    img1 = img[..., 1] # middle channel
    tmp_save_dir = os.path.dirname(im).replace("3ch", "1ch")
    os.makedirs(tmp_save_dir, exist_ok=True)
    status = cv2.imwrite(im.replace("3ch", "1ch"), img1)
