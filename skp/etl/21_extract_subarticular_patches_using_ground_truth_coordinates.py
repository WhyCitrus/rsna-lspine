import cv2
import numpy as np
import os
import pandas as pd


df = pd.read_csv("../../data/train_label_coordinates.csv")
df = df.loc[df.condition.apply(lambda x: "Subarticular" in x)]
df["level"] = df["level"].apply(lambda x: x.replace("/", "_"))
df