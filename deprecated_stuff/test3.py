import os
import cv2 as cv
import numpy as np
from ultralytics import YOLOv10 as YOLO
import some_utils
import datetime

val_path = "C:\\Users\\GPSERS-DEXTER-04\\Downloads\\cataovo-annotations\\YOLOv10-structure\\images\\val\\"
train_path = "C:\\Users\\GPSERS-DEXTER-04\\Downloads\\cataovo-annotations\\YOLOv10-structure\\images\\train\\"
evaluation = f"evaluation-{datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")}\\"
os.makedirs(evaluation, exist_ok=False)

model = YOLO("weights\\runs-200-s-shuffled\\last.pt")

val_imgs = list(map(lambda x: val_path + x, os.listdir(val_path)))
train_imgs = list(map(lambda x: train_path + x, os.listdir(train_path)))
all_imgs = val_imgs + train_imgs

img_batch = []

for img_path in val_imgs[:32]:
    img_batch.append(cv.imread(img_path))

# img_batch = np.array(img_batch)
model.predict(img_batch, conf=0.2)
# pip install -U ultralytics